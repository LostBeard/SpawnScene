// Run any ?autotest= page and stream its console back, with no gating of its own.
//
//   node _cdp_autotest.js "<query>" "<LogPrefix>"
//   node _cdp_autotest.js "autotest=dav3-pose&n=6" "Dav3Pose"
//
// Finishes on "[<Prefix>] DONE" or "[<Prefix>] FAIL". Exists because every gate so far has
// carried its own copy of this, and a one-off measurement should not need a bespoke harness.

const http = require('http');
const WebSocket = require('ws');
const { ensureChrome } = require('./_chrome_harness');

const QUERY = process.argv[2] || 'autotest=dav3-pose';
const PREFIX = process.argv[3] || 'Dav3Pose';
const MINUTES = parseInt(process.env.MINUTES || '30', 10);

let CDP = 9223;
const cdp = (p) => `http://127.0.0.1:${CDP}${p}`;
const get = (u) => new Promise((res, rej) =>
  http.get(u, r => { let d = ''; r.on('data', c => d += c); r.on('end', () => res(JSON.parse(d))); })
    .on('error', rej));
const closeTab = (id) => new Promise(res =>
  http.get(cdp('/json/close/' + id), r => { r.resume(); r.on('end', res); }).on('error', () => res()));

(async () => {
  const chrome = await ensureChrome();
  CDP = chrome.port;

  const before = new Set((await get(cdp('/json/list'))).map(t => t.id));
  const ver = await get(cdp('/json/version'));
  const bws = new WebSocket(ver.webSocketDebuggerUrl);
  await new Promise(r => bws.on('open', r));
  bws.send(JSON.stringify({ id: 1, method: 'Target.createTarget', params: { url: 'about:blank' } }));
  await new Promise(r => setTimeout(r, 800));
  bws.close();

  const tab = (await get(cdp('/json/list'))).find(t => t.type === 'page' && !before.has(t.id));
  if (!tab) throw new Error('could not open a tab');

  let closed = false;
  const cleanup = async () => { if (!closed) { closed = true; await closeTab(tab.id); } };
  process.on('SIGINT', async () => { await cleanup(); process.exit(130); });

  try {
    const ws = new WebSocket(tab.webSocketDebuggerUrl);
    await new Promise(r => ws.on('open', r));
    let id = 1;
    const pend = new Map();
    let done = false, failed = null;
    const doneRe = new RegExp(`\\[${PREFIX}\\] DONE`);
    const failRe = new RegExp(`\\[${PREFIX}\\] FAIL`);

    ws.on('message', raw => {
      const m = JSON.parse(raw.toString());
      if (m.id && pend.has(m.id)) pend.get(m.id)(m);
      if (m.method === 'Runtime.consoleAPICalled') {
        const s = (m.params.args || []).map(a => a.value ?? a.description ?? '').join(' ');
        console.log(s.slice(0, 400));
        if (doneRe.test(s)) done = true;
        if (failRe.test(s)) failed = s;
      }
    });
    const send = (method, params = {}) => new Promise((res, rej) => {
      const i = id++;
      const to = setTimeout(() => rej(new Error('timeout ' + method)), 3600000);
      pend.set(i, v => { clearTimeout(to); res(v); });
      ws.send(JSON.stringify({ id: i, method, params }));
    });

    await send('Runtime.enable');
    await send('Page.enable');
    const url = `http://127.0.0.1:8080/studio?${QUERY}&cb=${Date.now()}`;
    console.log(`\n=== ${QUERY} ===\n${url}\n`);
    await send('Page.navigate', { url });

    const deadline = Date.now() + MINUTES * 60 * 1000;
    while (Date.now() < deadline && !done && !failed) await new Promise(r => setTimeout(r, 400));

    if (failed) { console.log('\nFAILED'); process.exitCode = 1; }
    else if (!done) { console.log('\nTIMED OUT'); process.exitCode = 1; }
    else console.log('\nOK');
  } finally {
    await cleanup();
    await chrome.close();
  }
})().catch(e => { console.error(e); process.exit(1); });
