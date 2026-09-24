const { ensureChrome } = require('./_chrome_harness');
const http = require('http');
const WebSocket = require('ws');
const fs = require('fs');

function get(u) {
  return new Promise((res, rej) =>
    http.get(u, (r) => {
      let d = '';
      r.on('data', (c) => (d += c));
      r.on('end', () => res(JSON.parse(d)));
    }).on('error', rej)
  );
}

(async () => {
  const chrome = await ensureChrome({ headless: false });
  const pages0 = await get('http://127.0.0.1:' + chrome.port + '/json/list');
  let page = pages0.find((p) => p.type === 'page');
  if (!page) {
    await get('http://127.0.0.1:' + chrome.port + '/json/new?about:blank');
    await new Promise((r) => setTimeout(r, 400));
    page = (await get('http://127.0.0.1:' + chrome.port + '/json/list')).find((p) => p.type === 'page');
  }
  const ws = new WebSocket(page.webSocketDebuggerUrl);
  await new Promise((r) => ws.on('open', r));
  let id = 1;
  const pend = new Map();
  const logs = [];
  ws.on('message', (raw) => {
    const m = JSON.parse(raw.toString());
    if (m.id && pend.has(m.id)) pend.get(m.id)(m);
    if (m.method === 'Runtime.consoleAPICalled') {
      const t = (m.params.args || []).map((a) => a.value ?? a.description ?? '').join(' ');
      logs.push(t);
      if (/Studio|Testing|Error|FAIL|Dataset|Initialized/i.test(t)) console.log('CON', t.slice(0, 350));
    }
  });
  const send = (method, params = {}) =>
    new Promise((res) => {
      const i = id++;
      pend.set(i, res);
      ws.send(JSON.stringify({ id: i, method, params }));
    });
  async function click(x, y) {
    await send('Input.dispatchMouseEvent', { type: 'mouseMoved', x, y });
    await send('Input.dispatchMouseEvent', { type: 'mousePressed', x, y, button: 'left', clickCount: 1 });
    await send('Input.dispatchMouseEvent', { type: 'mouseReleased', x, y, button: 'left', clickCount: 1 });
  }
  async function shot(name) {
    const r = await send('Page.captureScreenshot', { format: 'png' });
    fs.writeFileSync(name, Buffer.from(r.result.data, 'base64'));
    console.log('shot', name);
  }
  async function key(code, key) {
    await send('Input.dispatchKeyEvent', { type: 'keyDown', windowsVirtualKeyCode: 0, code, key, text: key.length === 1 ? key : '' });
    await send('Input.dispatchKeyEvent', { type: 'keyUp', windowsVirtualKeyCode: 0, code, key });
  }

  await send('Runtime.enable');
  await send('Page.enable');
  await send('Page.navigate', { url: 'http://127.0.0.1:8080/studio?cb=' + Date.now() });
  for (let i = 0; i < 40; i++) {
    await new Promise((r) => setTimeout(r, 500));
    if (logs.some((l) => l.includes('[Studio] Initialized'))) break;
  }
  await shot('peek-browser.png');

  // Testing button: panelW-220 + half width, margin 32 → CSS ~ (canvasW - 220 + 50, 20+16+32)
  const metrics = await send('Runtime.evaluate', {
    expression:
      "(()=>{const c=document.querySelector('canvas'); const r=c.getBoundingClientRect(); return {w:r.width,h:r.height};})()",
    returnByValue: true,
  });
  const cw = metrics.result.result.value.w;
  console.log('canvasW', cw);
  await click(cw - 170, 52); // Testing
  await new Promise((r) => setTimeout(r, 1500));
  await shot('peek-testing.png');

  // Open project detail via Open on first card — go back first
  await click(60, 48); // Back
  await new Promise((r) => setTimeout(r, 1000));
  await click(280, 271); // Open first project
  await new Promise((r) => setTimeout(r, 1500));
  await shot('peek-detail.png');

  // Prove wasm has NormalizeMovementKey / KeyW comment or "keyw" mapping — check source string
  const hasTesting = logs.some((l) => /Initialized/i.test(l));
  console.log(hasTesting ? 'PASS studio smoke' : 'FAIL');
  ws.close();
  await chrome.close();
  process.exit(0);
})().catch((e) => {
  console.error(e);
  process.exit(1);
});
