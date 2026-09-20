// Passive listener: enables the Log domain (where Chrome reports WebGPU validation errors,
// which Runtime.consoleAPICalled does NOT carry) and dumps everything for N seconds.
const http = require('http');
const WebSocket = require('ws');
const SECONDS = parseInt(process.argv[2] || '120', 10);

function getJson(url) {
  return new Promise((resolve, reject) => {
    http.get(url, (res) => {
      let d = '';
      res.on('data', (c) => (d += c));
      res.on('end', () => resolve(JSON.parse(d)));
    }).on('error', reject);
  });
}

(async () => {
  const pages = (await getJson('http://127.0.0.1:9222/json/list'))
    .filter((p) => p.type === 'page' && p.url && p.url.includes('8080'));
  const page = pages.find((p) => p.url.includes('/studio')) || pages[0];
  if (!page) { console.log('no studio page'); process.exit(1); }

  const ws = new WebSocket(page.webSocketDebuggerUrl);
  let id = 1;
  const send = (method, params = {}) =>
    new Promise((r) => { ws.send(JSON.stringify({ id: id++, method, params })); setTimeout(r, 50); });

  ws.on('message', (raw) => {
    const m = JSON.parse(raw);
    if (m.method === 'Log.entryAdded') {
      const e = m.params.entry;
      // This is the channel WebGPU validation text arrives on.
      console.log(`[LOG ${e.level}/${e.source}] ${e.text}`);
    }
    if (m.method === 'Runtime.exceptionThrown') {
      const d = m.params.exceptionDetails;
      console.log(`[EXC] ${d.text}\n${d.exception?.description ?? ''}`);
    }
    if (m.method === 'Runtime.consoleAPICalled' && m.params.type === 'error') {
      const t = (m.params.args || []).map((a) => a.value ?? a.description ?? '').join(' ');
      console.log(`[CONSOLE.error] ${t}`);
    }
  });

  await new Promise((r) => ws.on('open', r));
  await send('Runtime.enable');
  await send('Log.enable');
  console.log(`listening ${SECONDS}s...`);
  await new Promise((r) => setTimeout(r, SECONDS * 1000));
  ws.close();
})();
