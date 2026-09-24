// Runs /studio?autotest=trainer-gate in an isolated Chrome and reports the result.
// One tab, closed in a finally; browser launched and killed by PID. Never the user's browser.
const http = require('http');
const path = require('path');
const WebSocket = require('ws');
const { ensureChrome, APP } = require('./_chrome_harness');

let CDP = 9223;
const cdp = p => `http://127.0.0.1:${CDP}${p}`;
const get = u => new Promise((res, rej) =>
  http.get(u, r => { let d = ''; r.on('data', c => d += c); r.on('end', () => res(JSON.parse(d))); }).on('error', rej));

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
  if (!tab) throw new Error('no tab');

  let ok = false;
  try {
    const ws = new WebSocket(tab.webSocketDebuggerUrl);
    await new Promise(r => ws.on('open', r));
    let id = 1; const pend = new Map(); const logs = [];
    ws.on('message', raw => {
      const m = JSON.parse(raw.toString());
      if (m.id && pend.has(m.id)) pend.get(m.id)(m);
      if (m.method === 'Runtime.consoleAPICalled') {
        const s = (m.params.args || []).map(a => a.value ?? a.description ?? '').join(' ');
        logs.push(s);
        if (/TrainerGate|Trainer\]|Error|error/i.test(s)) console.log('  CON', s.slice(0, 300));
      }
      if (m.method === 'Log.entryAdded') {
        const t = m.params.entry.text || '';
        if (/error|warn/i.test(m.params.entry.level)) console.log('  LOG', t.slice(0, 300));
      }
    });
    const send = (method, params = {}) => new Promise((res, rej) => {
      const i = id++; const to = setTimeout(() => rej(new Error('timeout ' + method)), 300000);
      pend.set(i, v => { clearTimeout(to); res(v); });
      ws.send(JSON.stringify({ id: i, method, params }));
    });

    await send('Runtime.enable');
    await send('Log.enable').catch(() => {});
    await send('Page.enable');
    await send('Page.navigate', { url: `${APP}/studio?autotest=trainer-gate&cb=${Date.now()}` });

    const deadline = Date.now() + 240000;
    while (Date.now() < deadline) {
      if (logs.some(l => /\[TrainerGate\] (PASS|FAIL)/.test(l))) break;
      await new Promise(r => setTimeout(r, 500));
    }
    const verdict = logs.find(l => /\[TrainerGate\] (PASS|FAIL)/.test(l));
    console.log('\nVERDICT:', verdict || '(timed out with no verdict)');
    ok = !!verdict && verdict.includes('PASS');
    ws.close();
  } finally {
    await new Promise(res => http.get(cdp('/json/close/' + tab.id), r => { r.resume(); r.on('end', res); }).on('error', () => res()));
    await chrome.close();
  }
  process.exit(ok ? 0 : 1);
})().catch(e => { console.error(e.message || e); process.exit(1); });
