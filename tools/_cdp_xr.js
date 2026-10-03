// WebXR in the harness's WebGPU Chrome with Meta's Immersive Web Emulator (a virtual Quest), no headset needed.
// Loads the Room sample (?autotest=generate-room&xrhook=1), waits for its PASS, enters an XR session from a user gesture
// (window.__spawnsceneEnterXR, the page's autotest hook), lets it run, and reports the XR log lines ([XRService], [XR]
// frame cost) and errors, then a screenshot of the page.
//
//   SPAWNSCENE_CHROME_EXTENSION=_scratch/iwe/ext SPAWNSCENE_CDP_PORT=9228 SPAWNSCENE_APP_PORT=8102 \
//     node tools/_cdp_xr.js [immersive-vr|immersive-ar] [out.png] [runMs]
const http = require('http');
const fs = require('fs');
const WebSocket = require('ws');
const { ensureChrome, APP } = require('./_chrome_harness');

const [mode = 'immersive-vr', out = '_shots/xr.png', runMs = '15000'] = process.argv.slice(2);
const get = u => new Promise((res, rej) =>
  http.get(u, r => { let d = ''; r.on('data', c => d += c); r.on('end', () => res(JSON.parse(d))); }).on('error', rej));

(async () => {
  const chrome = await ensureChrome();
  const cdp = p => `http://127.0.0.1:${chrome.port}${p}`;
  const before = new Set((await get(cdp('/json/list'))).map(t => t.id));
  const ver = await get(cdp('/json/version'));
  const bws = new WebSocket(ver.webSocketDebuggerUrl);
  await new Promise(r => bws.on('open', r));
  bws.send(JSON.stringify({ id: 1, method: 'Target.createTarget', params: { url: 'about:blank' } }));
  await new Promise(r => setTimeout(r, 800));
  bws.close();
  const tab = (await get(cdp('/json/list'))).find(t => t.type === 'page' && !before.has(t.id));
  if (!tab) throw new Error('no tab');

  const ws = new WebSocket(tab.webSocketDebuggerUrl);
  await new Promise(r => ws.on('open', r));
  let id = 1; const pend = new Map();
  let passed = false, hook = false;
  ws.on('message', raw => {
    const m = JSON.parse(raw.toString());
    if (m.id && pend.has(m.id)) pend.get(m.id)(m);
    if (m.method === 'Runtime.exceptionThrown') console.log('EXC ' + JSON.stringify(m.params.exceptionDetails).slice(0, 400));
    if (m.method === 'Runtime.consoleAPICalled') {
      const t = (m.params.args || []).map(a => a.value ?? a.description ?? '').join(' ');
      if (/\[Autotest\] PASS/.test(t)) passed = true;
      if (/\[Autotest\] XR hook ready/.test(t)) hook = true;
      if (/error|warn/.test(m.params.type) || /\[Autotest\]|\[XR|GPU ERROR|\[Studio\] (Entering|Failed|XR|immersive)/.test(t)) console.log('CON ' + t.slice(0, 300));
    }
  });
  const send = (method, params = {}) => new Promise(res => { const i = id++; pend.set(i, res); ws.send(JSON.stringify({ id: i, method, params })); });
  try {
    await send('Page.enable');
    await send('Runtime.enable');
    // The emulator as an extension needs --load-extension, which branded Chrome 151 ignores. Its page side is just
    // webxr-polyfill.js (a WebXR device in the page) initialised by a 'pa-device-init' event, so inject that directly,
    // with the Meta Quest Pro definition from its content.js, stereo on, 6 x 3 x 6 m room.
    const iwe = process.env.SPAWNSCENE_IWE || '_scratch/iwe/ext';
    const quest = {
      id: 'Meta Quest Pro', name: 'Meta Quest Pro', profile: 'meta-quest-touch-pro',
      modes: ['inline', 'immersive-vr', 'immersive-ar'], headset: { hasPosition: true, hasRotation: true },
      controllers: ['left', 'right'].map(h => ({ id: `Meta Quest Touch Pro (${h === 'left' ? 'Left' : 'Right'})`, buttonNum: 7,
        primaryButtonIndex: 1, primarySqueezeButtonIndex: 2, hasPosition: true, hasRotation: true, hasSqueezeButton: true, handedness: h })),
      polyfillInputMapping: { axes: [2, 3, 0, 1], buttons: [1, 2, null, 0, 3, 4, null] },
    };
    const init = `setTimeout(() => {
      window.dispatchEvent(new CustomEvent('pa-device-init', { detail: { deviceDefinition: ${JSON.stringify(quest)}, stereoEffect: true } }));
      window.dispatchEvent(new CustomEvent('pa-room-dimension-change', { detail: { dimension: { x: 6, y: 3, z: 6 } } }));
    }, 0);`;
    await send('Page.addScriptToEvaluateOnNewDocument', { source: fs.readFileSync(`${iwe}/dist/webxr-polyfill.js`, 'utf8') + ';' + init });
    await send('Emulation.setDeviceMetricsOverride', { width: 1600, height: 1000, deviceScaleFactor: 1, mobile: false });
    await send('Page.navigate', { url: `${APP}/studio?autotest=generate-room&render=stochastic&xrhook=1${process.env.SPAWNSCENE_XR_EXTRA || ''}` });
    const deadline = Date.now() + 10 * 60 * 1000;
    while (!(passed && hook) && Date.now() < deadline) await new Promise(r => setTimeout(r, 500));
    if (!(passed && hook)) throw new Error('the Room sample never passed / the XR hook never appeared');
    const xr = await send('Runtime.evaluate', {
      expression: "navigator.xr ? navigator.xr.isSessionSupported('" + mode + "') : Promise.resolve('no navigator.xr')",
      awaitPromise: true, returnByValue: true });
    console.log(`isSessionSupported('${mode}'): ${JSON.stringify(xr.result && xr.result.result && xr.result.result.value)}`);
    const r = await send('Runtime.evaluate', { expression: `window.__spawnsceneEnterXR('${mode}')`, userGesture: true });
    if (r.result && r.result.exceptionDetails) console.log('enter threw: ' + JSON.stringify(r.result.exceptionDetails).slice(0, 300));
    await new Promise(r2 => setTimeout(r2, parseInt(runMs, 10)));
    const png = await send('Page.captureScreenshot', { format: 'png' });
    fs.writeFileSync(out, Buffer.from(png.result.data, 'base64'));
    console.log('captured ' + out);
  } finally {
    try { ws.send(JSON.stringify({ id: id++, method: 'Page.close', params: {} })); } catch { }
    await new Promise(r => setTimeout(r, 300));
    try { ws.close(); } catch { }
    if (chrome.close) await chrome.close();
  }
})().then(() => process.exit(0), e => { console.error(e); process.exit(1); });
