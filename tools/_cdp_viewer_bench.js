// Viewer benchmark: one scene, the same camera poses and lens, every viewer - frame rate per pose and a screenshot per
// pose, in the harness's own WebGPU Chrome (run with SPAWNSCENE_CHROME_UNCAPPED=1 so frame rate = render cost).
//
//   node tools/_cdp_viewer_bench.js <viewer> <sceneUrl> <posesUrl> <outDir> [WxH=1600x900] [fov=50]
//     viewer: spark | gs3d | playcanvas (tools/viewer-bench/<viewer>.html, served next to the app)
//           | spawnscene (the app: ?import=<scene>&park=<pose>&fpslog=1, one load per pose)
//   Poses: tools/viewer-bench/make_poses.py. Writes <outDir>/<tag>_pose<k>.png and <outDir>/<tag>.json (tag = viewer,
//   or BENCH_TAG); BENCH_EXTRA=<query> is appended to the spawnscene URL (e.g. &lodpx=0 draws every splat).
const http = require('http');
const fs = require('fs');
const path = require('path');
const WebSocket = require('ws');
const { ensureChrome, APP } = require('./_chrome_harness');

const [viewer, sceneUrl, posesUrl, outDir, size = '1600x900', fovArg = '50'] = process.argv.slice(2);
const [W, H] = size.split('x').map(Number);
const fov = Number(fovArg);
const WARM = 4000, MEASURE = 8000;
const TAG = process.env.BENCH_TAG || viewer, EXTRA = process.env.BENCH_EXTRA || '';
const get = u => new Promise((res, rej) =>
  http.get(u, r => { let d = ''; r.on('data', c => d += c); r.on('end', () => res(JSON.parse(d))); }).on('error', rej));
const sleep = ms => new Promise(r => setTimeout(r, ms));

(async () => {
  fs.mkdirSync(outDir, { recursive: true });
  const poses = await get(`${APP}${posesUrl}`);
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
  let id = 1; const pend = new Map(); const waiters = [];
  const onLine = t => { for (const w of [...waiters]) if (w.re.test(t)) { waiters.splice(waiters.indexOf(w), 1); w.res(t); } };
  const fpsLog = [];
  ws.on('message', raw => {
    const m = JSON.parse(raw.toString());
    if (m.id && pend.has(m.id)) pend.get(m.id)(m);
    if (m.method === 'Runtime.exceptionThrown') {
      const d = m.params.exceptionDetails;
      console.log('EXC ' + ((d.exception && d.exception.description) || d.text || '').slice(0, 400));
    }
    if (m.method === 'Runtime.consoleAPICalled') {
      const t = (m.params.args || []).map(a => a.value ?? a.description ?? '').join(' ');
      if (/\[Bench\]|\[Import\]|error/i.test(t)) console.log('CON ' + t.slice(0, 240));
      const f = t.match(/^\[FPS\] ([0-9.]+)/);
      if (f) fpsLog.push({ t: Date.now(), fps: Number(f[1]) });
      onLine(t);
    }
  });
  const send = (method, params = {}) => new Promise(res => { const i = id++; pend.set(i, res); ws.send(JSON.stringify({ id: i, method, params })); });
  const waitFor = (re, ms) => new Promise((res, rej) => {
    const w = { re, res }; waiters.push(w);
    setTimeout(() => { const k = waiters.indexOf(w); if (k >= 0) { waiters.splice(k, 1); rej(new Error(`timeout waiting for ${re}`)); } }, ms);
  });
  const shot = async name => {
    const png = await send('Page.captureScreenshot', { format: 'png' });
    const file = path.join(outDir, name);
    fs.writeFileSync(file, Buffer.from(png.result.data, 'base64'));
    return file;
  };
  const results = [];
  try {
    await send('Page.enable'); await send('Runtime.enable');
    await send('Emulation.setDeviceMetricsOverride', { width: W, height: H, deviceScaleFactor: 1, mobile: false });
    if (viewer === 'spawnscene') {
      // One load per pose: &park seats the viewer at an exact camera, intrinsics included (fy from the vertical FOV).
      const fy = (H / 2) / Math.tan(fov * Math.PI / 360);
      for (let k = 0; k < poses.ours.length; k++) {
        const p = poses.ours[k];
        const fwd = p.target.map((v, i) => v - p.pos[i]); const len = Math.hypot(...fwd); const f = fwd.map(v => v / len);
        const park = [W, H, fy, fy, W / 2, H / 2, ...p.pos, ...f, ...p.up].map(v => +v.toFixed(5)).join(',');
        fpsLog.length = 0;
        const parked = waitFor(/\[Import\] PARKED/, 180000);
        await send('Page.navigate', { url: `${APP}/studio?import=${encodeURIComponent(sceneUrl)}&park=${park}&fpslog=1${EXTRA}` });
        await parked;
        await sleep(WARM);
        const t0 = Date.now(); await sleep(MEASURE);
        const window = fpsLog.filter(x => x.t > t0).map(x => x.fps);
        const fps = window.length ? window.reduce((a, b) => a + b, 0) / window.length : 0;
        const file = await shot(`${TAG}_pose${k}.png`);
        console.log(`[Bench] ${TAG} POSE ${k} fps ${fps.toFixed(1)} (${window.length} samples) -> ${file}`);
        results.push({ pose: k, fps });
      }
    } else {
      const url = `${APP}/bench/${viewer}.html?url=${encodeURIComponent(sceneUrl)}&poses=${encodeURIComponent(posesUrl)}` +
        `&fov=${fov}&warm=${WARM}&measure=${MEASURE}`;
      await send('Page.navigate', { url });
      for (let k = 0; k < poses.ply.length; k++) {
        const line = await waitFor(new RegExp(`\\[Bench\\] ${viewer} POSE ${k} READY`), 240000);
        const fps = Number(line.match(/fps ([0-9.]+)/)[1]);
        const file = await shot(`${viewer}_pose${k}.png`);
        console.log(`[Bench] ${viewer} POSE ${k} fps ${fps.toFixed(1)} -> ${file}`);
        results.push({ pose: k, fps });
        await send('Runtime.evaluate', { expression: 'window.__benchNext && window.__benchNext()' });
      }
    }
    fs.writeFileSync(path.join(outDir, `${TAG}.json`), JSON.stringify({ viewer, tag: TAG, extra: EXTRA, scene: sceneUrl, size, fov, results }, null, 1));
  } finally {
    try { ws.send(JSON.stringify({ id: id++, method: 'Page.close', params: {} })); } catch { }
    await sleep(300);
    try { ws.close(); } catch { }
    if (chrome.close) await chrome.close();
  }
})().then(() => process.exit(0), e => { console.error(e); process.exit(1); });
