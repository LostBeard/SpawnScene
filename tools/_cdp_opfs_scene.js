// Pull a saved scene out of a harness profile's OPFS into a .spawnscene file - no browser download. Chrome cancelled
// the download of a 737 MB export (Mip-NeRF 360 bicycle, 3.28M splats, 2026-10-04); this reads the project's scene
// files in 16 MB slices over CDP and writes the same layout SceneFile defines: "SPSCENE1", int32 header length, JSON
// header, packed splats, SH parts.
//
//   SPAWNSCENE_CDP_PORT=9233 SPAWNSCENE_APP_PORT=8104 node tools/_cdp_opfs_scene.js <project name substring> <out.spawnscene>
const http = require('http');
const fs = require('fs');
const WebSocket = require('ws');
const { ensureChrome, APP } = require('./_chrome_harness');

const [match, out] = process.argv.slice(2);
if (!match || !out) { console.error('usage: _cdp_opfs_scene.js <project name substring> <out.spawnscene>'); process.exit(2); }
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
  // The app's origin: a static page is enough, OPFS is per origin.
  bws.send(JSON.stringify({ id: 1, method: 'Target.createTarget', params: { url: `${APP}/favicon.ico` } }));
  await sleep(1500);
  bws.close();
  const tab = (await get(cdp('/json/list'))).find(t => t.type === 'page' && !before.has(t.id));
  const ws = new WebSocket(tab.webSocketDebuggerUrl);
  await new Promise(r => ws.on('open', r));
  let id = 1; const pend = new Map();
  ws.on('message', raw => { const m = JSON.parse(raw.toString()); if (m.id && pend.has(m.id)) pend.get(m.id)(m); });
  const send = (method, params = {}) => new Promise(res => { const i = id++; pend.set(i, res); ws.send(JSON.stringify({ id: i, method, params })); });
  const evalJs = async expr => {
    const r = await send('Runtime.evaluate', { expression: expr, awaitPromise: true, returnByValue: true });
    if (r.result.exceptionDetails) throw new Error(JSON.stringify(r.result.exceptionDetails).slice(0, 400));
    return r.result.result.value;
  };
  try {
    const dirJs = `(async () => { const root = await navigator.storage.getDirectory(); return await root.getDirectoryHandle('spawnscene'); })()`;
    const index = JSON.parse(await evalJs(`(async () => { const d = await ${dirJs}; const f = await (await d.getFileHandle('projects.json')).getFile(); return await f.text(); })()`));
    const projects = Array.isArray(index) ? index : (index.Projects || index.projects || []);
    const p = projects.filter(x => (x.Name || x.name || '').includes(match)).sort((a, b) => String(b.ModifiedAt || b.modifiedAt).localeCompare(String(a.ModifiedAt || a.modifiedAt)))[0];
    if (!p) throw new Error(`no project matching '${match}' in ${projects.map(x => x.Name || x.name).join(', ')}`);
    const scenes = p.Scenes || p.scenes;
    const s = scenes.sort((a, b) => String(b.CreatedAt || b.createdAt).localeCompare(String(a.CreatedAt || a.createdAt)))[0];
    const pid = p.Id || p.id, sid = s.Id || s.id;
    const shParts = s.ShParts ?? s.shParts ?? 0;
    console.log(`project '${p.Name || p.name}' (${pid}), scene ${sid}: ${s.SplatCount ?? s.splatCount} splats, SH degree ${s.ShDegree ?? s.shDegree}, ${shParts} parts`);
    const header = {
      Name: p.Name || p.name, SplatCount: s.SplatCount ?? s.splatCount, FloatsPerSplat: s.FloatsPerSplat ?? s.floatsPerSplat ?? 14,
      ColoursAreShDc: s.ColoursAreShDc ?? s.coloursAreShDc ?? false, ShDegree: shParts > 0 ? (s.ShDegree ?? s.shDegree) : 0,
      ShParts: shParts, TrainedIterations: s.TrainedIterations ?? s.trainedIterations ?? 0, SavedAt: new Date().toISOString(),
      HomeView: s.HomeView ?? s.homeView ?? null,
    };
    const hj = Buffer.from(JSON.stringify(header));
    const fd = fs.openSync(out, 'w');
    const lenBuf = Buffer.alloc(4); lenBuf.writeInt32LE(hj.length);
    fs.writeSync(fd, Buffer.from('SPSCENE1')); fs.writeSync(fd, lenBuf); fs.writeSync(fd, hj);
    const files = [`${sid}.bin`, ...Array.from({ length: shParts }, (_, k) => `${sid}.sh${k}.bin`)];
    const CHUNK = 16 * 1024 * 1024;
    for (const name of files) {
      const fileJs = `(async () => { const d = await ${dirJs}; const pd = await (await d.getDirectoryHandle('projects')).getDirectoryHandle('${pid}'); const sd = await pd.getDirectoryHandle('scenes'); return await (await sd.getFileHandle('${name}')).getFile(); })()`;
      const size = await evalJs(`(async () => (await ${fileJs}).size)()`);
      for (let off = 0; off < size; off += CHUNK) {
        const b64 = await evalJs(`(async () => { const f = await ${fileJs}; const b = new Uint8Array(await f.slice(${off}, ${Math.min(size, off + CHUNK)}).arrayBuffer());
          let s = ''; for (let i = 0; i < b.length; i += 32768) s += String.fromCharCode.apply(null, b.subarray(i, i + 32768)); return btoa(s); })()`);
        fs.writeSync(fd, Buffer.from(b64, 'base64'));
      }
      console.log(`  ${name}: ${(size / 1048576).toFixed(0)} MB`);
    }
    fs.closeSync(fd);
    console.log(`wrote ${out} (${(fs.statSync(out).size / 1048576).toFixed(0)} MB)`);
  } finally {
    try { await send('Page.close'); } catch { }
    try { ws.close(); } catch { }
    if (chrome.close) await chrome.close();
  }
})().then(() => process.exit(0), e => { console.error(e); process.exit(1); });
