// ORT-web LightGlue benchmark driver (2026-10-01): serves _scratch/ortbench on BENCH_PORT (cross-origin isolated, so ORT's
// WASM backend can thread), opens it in an isolated Chrome (tools/_chrome_harness.js; SPAWNSCENE_CDP_PORT picks the debug
// port), prints the page's "[ortbench]" console lines, and closes the Chrome it started.
// Setup: _scratch/ortbench/ holds onnxruntime-web's dist files (ort.all.min.mjs + ort-wasm-simd-threaded.*), the Kornia
// matcher (_scratch/kornia/lightglue_matcher_k1024.onnx) and its reference inputs (_scratch/kornia/ref/k1024_{nkp,desc}.f32);
// the page itself is copied from tools/ort_lightglue_bench.html. Edit the page's [ep, opt] list to choose backends.
// MEASURED 2026-10-01 (RTX 4070, Chrome 151): onnxruntime-web 1.30.0 WebGPU FAILS on this model at every optimization level
// (Reshape {1,1,1024,256} -> {2,1024,256}; the 1.22 dev build throws a bare exception); WASM 12 threads ~1,080 ms a pair,
// 1 thread ~2,900 ms. SpawnDev.ILGPU.ML on WebGPU, in SpawnScene: ~550-590 ms a pair.
// Usage: SPAWNSCENE_CDP_PORT=9228 node tools/ort_lightglue_bench.js
const http = require('http');
const path = require('path');
const { spawn } = require('child_process');
const WebSocket = require('ws');
const { ensureChrome } = require('./_chrome_harness');

const BENCH_PORT = parseInt(process.env.BENCH_PORT || '8111', 10);
const root = path.resolve(__dirname, '..', '_scratch', 'ortbench');

(async () => {
  // Its own static server with COOP/COEP, so the page is cross-origin isolated and ORT's WASM backend can use threads.
  const fs = require('fs');
  fs.copyFileSync(path.join(__dirname, 'ort_lightglue_bench.html'), path.join(root, 'index.html'));
  const types = { '.html': 'text/html', '.mjs': 'text/javascript', '.js': 'text/javascript', '.wasm': 'application/wasm' };
  const srv = http.createServer((req, res) => {
    const file = path.join(root, decodeURIComponent(req.url.split('?')[0]).replace(/^\/+/, '') || 'index.html');
    fs.readFile(file, (err, data) => {
      if (err) { res.writeHead(404); res.end(); return; }
      res.writeHead(200, { 'Content-Type': types[path.extname(file)] || 'application/octet-stream',
        'Cross-Origin-Opener-Policy': 'same-origin', 'Cross-Origin-Embedder-Policy': 'require-corp' });
      res.end(data);
    });
  }).listen(BENCH_PORT);
  const server = { pid: -1 };
  const chrome = await ensureChrome();
  try {
    const target = await new Promise((res, rej) => {
      const req = http.request({ host: '127.0.0.1', port: chrome.port, method: 'PUT',
        path: `/json/new?http://127.0.0.1:${BENCH_PORT}/index.html` }, r => { let d = ''; r.on('data', c => d += c); r.on('end', () => res(JSON.parse(d))); });
      req.on('error', rej); req.end();
    });
    const ws = new WebSocket(target.webSocketDebuggerUrl);
    await new Promise(r => ws.on('open', r));
    ws.send(JSON.stringify({ id: 1, method: 'Runtime.enable' }));
    await new Promise((resolve) => {
      const timer = setTimeout(() => { console.log('[bench] TIMEOUT'); resolve(); }, 10 * 60 * 1000);
      ws.on('message', (data) => {
        const m = JSON.parse(data);
        if (m.method !== 'Runtime.consoleAPICalled') return;
        const text = m.params.args.map(a => a.value ?? a.description ?? '').join(' ');
        if (!text.startsWith('[ortbench]')) { if (process.env.BENCH_ALL && /error|fail|not supported|unsupported|exception/i.test(text)) console.log('  console: ' + text.slice(0, 400)); return; }
        console.log(text);
        if (text.includes('DONE')) { clearTimeout(timer); resolve(); }
      });
    });
    ws.close();
  } finally {
    await chrome.close();
    srv.close();
  }
})().catch(e => { console.error(e); process.exit(1); });
