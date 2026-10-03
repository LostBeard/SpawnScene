// Scene editing in the desktop viewer, driven with real mouse events (CDP Input.dispatchMouseEvent): loads the Room
// sample, opens Edit, selects with a dragged rectangle, then Delete -> Undo -> Keep only, capturing each step and
// printing the [Edit] log lines (selected counts, undo depth).
//
//   SPAWNSCENE_CDP_PORT=9228 SPAWNSCENE_APP_PORT=8102 node tools/_cdp_edit.js [outPrefix]
const http = require('http');
const fs = require('fs');
const WebSocket = require('ws');
const { ensureChrome, APP } = require('./_chrome_harness');

const [prefix = '_shots/edit'] = process.argv.slice(2);
const get = u => new Promise((res, rej) =>
  http.get(u, r => { let d = ''; r.on('data', c => d += c); r.on('end', () => res(JSON.parse(d))); }).on('error', rej));
const sleep = ms => new Promise(r => setTimeout(r, ms));

(async () => {
  const chrome = await ensureChrome();
  const cdp = p => `http://127.0.0.1:${chrome.port}${p}`;
  const before = new Set((await get(cdp('/json/list'))).map(t => t.id));
  const ver = await get(cdp('/json/version'));
  const bws = new WebSocket(ver.webSocketDebuggerUrl);
  await new Promise(r => bws.on('open', r));
  bws.send(JSON.stringify({ id: 1, method: 'Target.createTarget', params: { url: 'about:blank' } }));
  await sleep(800);
  bws.close();
  const tab = (await get(cdp('/json/list'))).find(t => t.type === 'page' && !before.has(t.id));
  if (!tab) throw new Error('no tab');

  const ws = new WebSocket(tab.webSocketDebuggerUrl);
  await new Promise(r => ws.on('open', r));
  let id = 1; const pend = new Map(); let passed = false;
  ws.on('message', raw => {
    const m = JSON.parse(raw.toString());
    if (m.id && pend.has(m.id)) pend.get(m.id)(m);
    if (m.method === 'Runtime.exceptionThrown') console.log('EXC ' + JSON.stringify(m.params.exceptionDetails).slice(0, 300));
    if (m.method === 'Runtime.consoleAPICalled') {
      const t = (m.params.args || []).map(a => a.value ?? a.description ?? '').join(' ');
      if (/\[Autotest\] PASS/.test(t)) passed = true;
      if (/error/.test(m.params.type) || /\[Autotest\]|\[Edit\]|GPU ERROR/.test(t)) console.log('CON ' + t.slice(0, 300));
    }
  });
  const send = (method, params = {}) => new Promise(res => { const i = id++; pend.set(i, res); ws.send(JSON.stringify({ id: i, method, params })); });
  const shot = async name => {
    const png = await send('Page.captureScreenshot', { format: 'png' });
    fs.writeFileSync(`${prefix}_${name}.png`, Buffer.from(png.result.data, 'base64'));
    console.log(`captured ${prefix}_${name}.png`);
  };
  const touch = (type, pts) => send('Input.dispatchTouchEvent', { type, touchPoints: pts.map(([x, y], i) => ({ x, y, id: i })) });
  // A gesture as a sequence of finger sets, one frame apart.
  const gesture = async frames => {
    await touch('touchStart', frames[0]);
    for (const f of frames.slice(1)) { await sleep(33); await touch('touchMove', f); }
    await sleep(33);
    await touch('touchEnd', []);
  };
  const mouse = (type, x, y) => send('Input.dispatchMouseEvent', { type, x, y, button: 'left', buttons: type === 'mouseReleased' ? 0 : 1, clickCount: 1 });
  const click = async (x, y) => { await send('Input.dispatchMouseEvent', { type: 'mouseMoved', x, y }); await sleep(60); await mouse('mousePressed', x, y); await sleep(80); await mouse('mouseReleased', x, y); await sleep(300); };
  const drag = async (x0, y0, x1, y1) => {
    await send('Input.dispatchMouseEvent', { type: 'mouseMoved', x: x0, y: y0 }); await sleep(60);
    await mouse('mousePressed', x0, y0);
    for (let k = 1; k <= 10; k++) { await sleep(33); await send('Input.dispatchMouseEvent', { type: 'mouseMoved', x: x0 + (x1 - x0) * k / 10, y: y0 + (y1 - y0) * k / 10, buttons: 1 }); }
    await sleep(60); await mouse('mouseReleased', x1, y1); await sleep(600);
  };
  try {
    await send('Page.enable');
    await send('Runtime.enable');
    await send('Emulation.setDeviceMetricsOverride', { width: 1600, height: 1000, deviceScaleFactor: 1, mobile: false });
    await send('Page.navigate', { url: `${APP}/studio?autotest=generate-room&render=stochastic` });
    const deadline = Date.now() + 10 * 60 * 1000;
    while (!passed && Date.now() < deadline) await sleep(500);
    if (!passed) throw new Error('the Room sample never passed');
    await sleep(1500);
    // Top bar: Edit sits left of AR (canvas 1600 wide, Send to Headset hidden on loopback). Toolbar: left edge.
    await click(1208, 28); await sleep(500);
    await click(90, 97);                         // Select
    await drag(600, 520, 1000, 800);             // over the sofa and table
    await shot('0_selected');
    await click(90, 139); await sleep(1500);     // Delete
    await shot('1_deleted');
    await click(90, 223); await sleep(1500);     // Undo
    await shot('2_undone');
    await drag(600, 520, 1000, 800);             // select again (still in Select mode)
    await click(90, 181); await sleep(1500);     // Keep only
    await shot('3_kept');
  } finally {
    try { ws.send(JSON.stringify({ id: id++, method: 'Page.close', params: {} })); } catch { }
    await sleep(300);
    try { ws.close(); } catch { }
    if (chrome.close) await chrome.close();
  }
})().then(() => process.exit(0), e => { console.error(e); process.exit(1); });
