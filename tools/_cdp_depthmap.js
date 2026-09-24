// Dump a depth map to PNG so two configurations can be compared like for like.
//
//   node _cdp_depthmap.js <tag> [imageUrl] [patchesPerSide] [disparity]
//
// Writes _shots/depth/<tag>.png from canvas #depthdump, which the app fills from the RAW float
// depth using the same turbo mapping it displays with. No screenshots, no JPEG, no crops -
// two runs differ only in what is being tested.

const http = require('http');
const fs = require('fs');
const path = require('path');
const WebSocket = require('ws');
const { ensureChrome, APP } = require('./_chrome_harness');

const TAG = process.argv[2] || 'depth';
const IMG = process.argv[3] || 'samples/living-room-hd-2.jpg';
const PATCHES = process.argv[4] || '37';
const DISP = process.argv[5] === '1' ? '&disparity=1' : '';

let CDP = 9223;
const cdp = (p) => `http://127.0.0.1:${CDP}${p}`;
const get = (u) => new Promise((res, rej) =>
  http.get(u, r => { let d=''; r.on('data',c=>d+=c); r.on('end',()=>res(JSON.parse(d))); }).on('error', rej));
const closeTab = (id) => new Promise(res =>
  http.get(cdp('/json/close/'+id), r => { r.resume(); r.on('end', res); }).on('error', () => res()));

(async () => {
  const chrome = await ensureChrome();
  CDP = chrome.port;
  const before = new Set((await get(cdp('/json/list'))).map(t => t.id));
  const ver = await get(cdp('/json/version'));
  const bws = new WebSocket(ver.webSocketDebuggerUrl);
  await new Promise(r => bws.on('open', r));
  bws.send(JSON.stringify({ id:1, method:'Target.createTarget', params:{ url:'about:blank' } }));
  await new Promise(r => setTimeout(r, 800));
  bws.close();
  const tab = (await get(cdp('/json/list'))).find(t => t.type==='page' && !before.has(t.id));
  if (!tab) throw new Error('could not open a tab');

  let closed = false;
  const cleanup = async () => { if (!closed) { closed = true; await closeTab(tab.id); } };
  process.on('SIGINT', async () => { await cleanup(); process.exit(130); });

  try {
    const ws = new WebSocket(tab.webSocketDebuggerUrl);
    await new Promise(r => ws.on('open', r));
    let id = 1; const pend = new Map();
    let ready = false, failed = null;
    ws.on('message', raw => {
      const m = JSON.parse(raw.toString());
      if (m.id && pend.has(m.id)) pend.get(m.id)(m);
      if (m.method === 'Runtime.consoleAPICalled') {
        const s = (m.params.args||[]).map(a => a.value ?? a.description ?? '').join(' ');
        if (/DepthMap|Depth\]|FAIL|Error/i.test(s)) console.log('  ', s.slice(0,200));
        if (/\[DepthMap\] READY/.test(s)) ready = true;
        if (/\[DepthMap\] FAIL/.test(s)) failed = s;
      }
    });
    const send = (method, params={}) => new Promise((res, rej) => {
      const i = id++;
      const to = setTimeout(() => rej(new Error('timeout '+method)), 900000);
      pend.set(i, v => { clearTimeout(to); res(v); });
      ws.send(JSON.stringify({ id:i, method, params }));
    });

    await send('Runtime.enable');
    await send('Page.enable');
    const url = `${APP}/studio?autotest=depthmap&img=${encodeURIComponent(IMG)}`
              + `&patches=${PATCHES}${DISP}&cb=${Date.now()}`;
    console.log(`\n=== ${TAG}: ${IMG} @ ${PATCHES}x${PATCHES} patches ===`);
    await send('Page.navigate', { url });

    const deadline = Date.now() + 15*60*1000;
    while (Date.now() < deadline && !ready && !failed) await new Promise(r => setTimeout(r, 400));
    if (failed) { console.log('FAILED'); process.exitCode = 1; return; }
    if (!ready) { console.log('TIMED OUT'); process.exitCode = 1; return; }

    const res = await send('Runtime.evaluate', {
      expression: `document.getElementById('depthdump').toDataURL('image/png')`,
      returnByValue: true,
    });
    const dataUrl = res.result.result.value;
    if (!dataUrl || !dataUrl.startsWith('data:image/png')) throw new Error('no canvas data');
    const out = path.join(__dirname, '..', '_shots', 'depth', `${TAG}.png`);
    fs.mkdirSync(path.dirname(out), { recursive: true });
    fs.writeFileSync(out, Buffer.from(dataUrl.split(',')[1], 'base64'));
    console.log('wrote ' + out);
  } finally {
    await cleanup();
    await chrome.close();
  }
})().catch(e => { console.error(e); process.exit(1); });
