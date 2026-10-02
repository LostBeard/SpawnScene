// Screenshot any app page in the harness's own WebGPU Chrome (headless Chrome without a GPU never gets past
// "Initializing GPU..."). One tab, closed in a finally; the browser is the harness's (launched and killed by PID).
//
//   node tools/_cdp_page.js <path> <out.png> [WxH] [waitMs]
//   e.g. SPAWNSCENE_APP_PORT=8102 SPAWNSCENE_CDP_PORT=9229 node tools/_cdp_page.js / _shots/home.png 1600x1000 6000
const http = require('http');
const fs = require('fs');
const WebSocket = require('ws');
const { ensureChrome, APP } = require('./_chrome_harness');

const [pagePath = '/', out = 'page.png', size = '1600x1000', waitMs = '6000'] = process.argv.slice(2);
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
  ws.on('message', raw => {
    const m = JSON.parse(raw.toString());
    if (m.id && pend.has(m.id)) pend.get(m.id)(m);
    // Errors and exceptions while the page loads (a blank page usually says why here).
    if (m.method === 'Runtime.exceptionThrown') console.log('EXC ' + JSON.stringify(m.params.exceptionDetails).slice(0, 400));
    if (m.method === 'Runtime.consoleAPICalled') {
      const t = (m.params.args || []).map(a => a.value ?? a.description ?? '').join(' ');
      // Errors and warnings, and the app's autotest verdicts (?autotest=... pages report PASS / FAIL on the console).
      if (/error|warn/.test(m.params.type) || /\[Autotest\]/.test(t)) console.log('CON ' + t.slice(0, 300));
    }
  });
  const send = (method, params = {}) => new Promise(res => { const i = id++; pend.set(i, res); ws.send(JSON.stringify({ id: i, method, params })); });
  try {
    const [w, h] = size.split('x').map(Number);
    await send('Page.enable');
    await send('Runtime.enable');
    await send('Emulation.setDeviceMetricsOverride', { width: w, height: h, deviceScaleFactor: 1, mobile: false });
    await send('Page.navigate', { url: `${APP}${pagePath}` });
    await new Promise(r => setTimeout(r, parseInt(waitMs, 10)));
    // What the page holds, so a blank capture can be told apart from an empty page.
    const txt = await send('Runtime.evaluate', { expression: "JSON.stringify({text: document.body.innerText.slice(0, 160), " +
      "opacity: [...document.querySelectorAll('.animate-in')].map(e => getComputedStyle(e).opacity).join(','), " +
      "hidden: document.visibilityState, href: location.href, html: document.body.innerHTML.length, app: (document.querySelector('#app')||{}).innerHTML?.slice(0,200)})", returnByValue: true });
    console.log('page: ' + (txt.result && txt.result.result && txt.result.result.value));
    const png = await send('Page.captureScreenshot', { format: 'png' });
    fs.writeFileSync(out, Buffer.from(png.result.data, 'base64'));
    console.log('captured ' + out);
  } finally {
    // Page.close is fire-and-forget: closing the page drops this socket before any reply comes back, so awaiting it hung
    // the process forever after the capture (the stochastic gate's calls timed out at 600 s, 2026-10-02).
    try { ws.send(JSON.stringify({ id: id++, method: 'Page.close', params: {} })); } catch { }
    await new Promise(r => setTimeout(r, 300));
    try { ws.close(); } catch { }
    // The harness handle's close() shuts down the Chrome this run launched (it attaches to nothing it did not start).
    if (chrome.close) await chrome.close();
  }
})().then(() => process.exit(0), e => { console.error(e); process.exit(1); });
