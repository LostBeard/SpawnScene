// Probe how this Chrome gives up video frames: seek + drawImage, seek + requestVideoFrameCallback, and
// createImageBitmap(video), each at t=0 and t=6 s of datasets/TruckVideo/truck.mp4. Prints a pixel signature per
// method and time; a method that returns the same signature for both times is not delivering the seeked frame.
const http = require('http');
const WebSocket = require('ws');
const { ensureChrome, APP } = require('./_chrome_harness');

const get = u => new Promise((res, rej) =>
  http.get(u, r => { let d = ''; r.on('data', c => d += c); r.on('end', () => res(JSON.parse(d))); }).on('error', rej));

const PROBE = `(async () => {
  const out = [];
  const v = document.createElement('video');
  v.muted = true; v.playsInline = true; v.preload = 'auto';
  v.style.cssText = 'position:fixed;left:0;top:0;width:320px;height:180px;z-index:9999';
  document.body.appendChild(v);
  const blob = await (await fetch('datasets/TruckVideo/truck.mp4')).blob(); v.src = URL.createObjectURL(blob);
  await new Promise((res, rej) => { v.addEventListener('loadeddata', res, { once: true }); v.addEventListener('error', () => rej(new Error('load')), { once: true }); });
  out.push('duration ' + v.duration + ' size ' + v.videoWidth + 'x' + v.videoHeight + ' rVFC ' + !!v.requestVideoFrameCallback);
  const c = new OffscreenCanvas(64, 36), ctx = c.getContext('2d', { willReadFrequently: true });
  const sig = () => { const d = ctx.getImageData(0, 0, 64, 36).data; let a = 0; for (let i = 0; i < d.length; i += 4) a = (a * 31 + d[i]) >>> 0; return a; };
  const seeked = t => new Promise(r => { v.addEventListener('seeked', () => r(), { once: true }); v.currentTime = t; });
  const frame = () => new Promise(r => { let done = false; const f = () => { if (!done) { done = true; r(); } }; if (v.requestVideoFrameCallback) v.requestVideoFrameCallback(f); setTimeout(f, 1500); });
  for (const t of [0.05, 6.05]) {
    await seeked(t);
    ctx.drawImage(v, 0, 0, 64, 36); const a = sig();
    await frame();
    ctx.drawImage(v, 0, 0, 64, 36); const b = sig();
    const bmp = await createImageBitmap(v); ctx.drawImage(bmp, 0, 0, 64, 36); const cc = sig(); bmp.close();
    let vf = 'n/a';
    if (window.VideoFrame) { try { const f = new VideoFrame(v); ctx.drawImage(f, 0, 0, 64, 36); vf = sig(); f.close(); } catch (e) { vf = 'err ' + e.message; } }
    out.push('t=' + t + ' currentTime=' + v.currentTime.toFixed(3) + ' seek+draw ' + a + ' | +rVFC ' + b + ' | createImageBitmap ' + cc + ' | VideoFrame ' + vf);
  }
  let t0 = performance.now();
  for (let i = 1; i <= 20; i++) await seeked(i * 0.6);
  out.push('20 seeks, no frame wait: ' + (performance.now() - t0).toFixed(0) + ' ms');
  t0 = performance.now(); let timeouts = 0;
  for (let i = 1; i <= 20; i++) { await seeked(i * 0.6 + 0.03); const t1 = performance.now(); await frame(); if (performance.now() - t1 > 1400) timeouts++; }
  out.push('20 seeks + rVFC wait: ' + (performance.now() - t0).toFixed(0) + ' ms, ' + timeouts + ' timed out');
  v.remove();
  return out.join('\\n');
})()`;

(async () => {
  process.env.SPAWNSCENE_CDP_PORT = process.env.SPAWNSCENE_CDP_PORT || '9235';
  const chrome = await ensureChrome();
  const cdp = p => `http://127.0.0.1:${chrome.port}${p}`;
  const ver = await get(cdp('/json/version'));
  const before = new Set((await get(cdp('/json/list'))).map(t => t.id));
  const bws = new WebSocket(ver.webSocketDebuggerUrl);
  await new Promise(r => bws.on('open', r));
  bws.send(JSON.stringify({ id: 1, method: 'Target.createTarget', params: { url: APP + '/' } }));
  await new Promise(r => setTimeout(r, 1500));
  bws.close();
  const tab = (await get(cdp('/json/list'))).find(t => t.type === 'page' && !before.has(t.id));
  const ws = new WebSocket(tab.webSocketDebuggerUrl);
  await new Promise(r => ws.on('open', r));
  let id = 1; const pend = new Map();
  ws.on('message', raw => { const m = JSON.parse(raw.toString()); if (m.id && pend.has(m.id)) pend.get(m.id)(m); });
  const send = (method, params = {}) => new Promise(res => { const i = id++; pend.set(i, res); ws.send(JSON.stringify({ id: i, method, params })); });
  await new Promise(r => setTimeout(r, 3000));
  const r = await send('Runtime.evaluate', { expression: PROBE, awaitPromise: true, returnByValue: true });
  console.log(r.result.exceptionDetails ? JSON.stringify(r.result.exceptionDetails).slice(0, 500) : r.result.result.value);
  ws.close();
  await chrome.close();
})().catch(e => { console.error(e); process.exit(1); });
