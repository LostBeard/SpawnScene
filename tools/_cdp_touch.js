// Touch navigation in the desktop viewer, driven with real touch events (CDP Input.dispatchTouchEvent): loads the
// Room sample (?autotest=generate-room), waits for its PASS, then captures the view before, after a one-finger drag
// (look) and after a two-finger pinch out (dolly in).
//
//   SPAWNSCENE_CDP_PORT=9228 SPAWNSCENE_APP_PORT=8102 node tools/_cdp_touch.js [outPrefix]
const http = require('http');
const fs = require('fs');
const WebSocket = require('ws');
const { ensureChrome, APP } = require('./_chrome_harness');

const [prefix = '_shots/touch'] = process.argv.slice(2);
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
      if (passed || /error/.test(m.params.type) || /\[Autotest\]|GPU ERROR/.test(t)) console.log('CON ' + t.slice(0, 300));
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
  try {
    await send('Page.enable');
    await send('Runtime.enable');
    await send('Emulation.setDeviceMetricsOverride', { width: 1600, height: 1000, deviceScaleFactor: 1, mobile: false });
    await send('Emulation.setTouchEmulationEnabled', { enabled: true, maxTouchPoints: 5 });
    await send('Page.navigate', { url: `${APP}/studio?autotest=generate-room&render=stochastic` });
    const deadline = Date.now() + 10 * 60 * 1000;
    while (!passed && Date.now() < deadline) await sleep(500);
    if (!passed) throw new Error('the Room sample never passed');
    await sleep(1500);
    if (process.env.SPAWNSCENE_TOUCH_TRACE)
      await send('Runtime.evaluate', { expression: `addEventListener('touchmove', e => console.log('[TouchTrace] ' +
        Array.from(e.touches).map(t => t.identifier + '@' + t.clientX + ',' + t.clientY).join(' ') + ' | changed ' +
        Array.from(e.changedTouches).map(t => t.identifier + '@' + t.clientX + ',' + t.clientY).join(' ')), true)` });
    await shot('0_start');
    // One finger, 200 px to the right across the middle of the view: the scene should follow it (view turns left).
    await gesture(Array.from({ length: 11 }, (_, k) => [[800 + 20 * k, 500]]));
    await sleep(1200);
    await shot('1_drag_right');
    // Two fingers spreading from 100 to 400 px apart about the centre: dolly in.
    await gesture(Array.from({ length: 11 }, (_, k) => [[750 - 15 * k, 500], [850 + 15 * k, 500]]));
    await sleep(1200);
    await shot('2_pinch_out');
    await sleep(3000);
    await shot('3_after');
  } finally {
    try { ws.send(JSON.stringify({ id: id++, method: 'Page.close', params: {} })); } catch { }
    await sleep(300);
    try { ws.close(); } catch { }
    if (chrome.close) await chrome.close();
  }
})().then(() => process.exit(0), e => { console.error(e); process.exit(1); });
