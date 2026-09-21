// Screenshot whatever the harness browser is currently showing, without touching it.
//
// Read-only on purpose: no navigation, no input, no reload. For looking at a run that is
// already going, especially when someone watching it reports something the logs do not show.
const http = require('http');
const WebSocket = require('ws');

const PORT = process.env.CDP_PORT || 9223;
const OUT = process.argv[2] || 'peek.png';

const get = (u) => new Promise((res, rej) =>
  http.get(u, r => { let d=''; r.on('data',c=>d+=c); r.on('end',()=>res(JSON.parse(d))); }).on('error', rej));

(async () => {
  const tabs = await get(`http://127.0.0.1:${PORT}/json/list`);
  const page = tabs.find(t => t.type === 'page' && /studio/.test(t.url || ''));
  if (!page) { console.log('no studio page open on ' + PORT); process.exit(1); }
  console.log('peeking at: ' + page.url);

  const ws = new WebSocket(page.webSocketDebuggerUrl);
  await new Promise(r => ws.on('open', r));
  let id = 1;
  const pend = new Map();
  ws.on('message', raw => {
    const m = JSON.parse(raw.toString());
    if (m.id && pend.has(m.id)) pend.get(m.id)(m);
  });
  const send = (method, params={}) => new Promise(res => {
    const i = id++; pend.set(i, res);
    ws.send(JSON.stringify({ id: i, method, params }));
  });

  const shot = await send('Page.captureScreenshot', { format: 'png' });
  require('fs').writeFileSync(OUT, Buffer.from(shot.result.data, 'base64'));
  console.log('wrote ' + OUT);
  ws.close();
})().catch(e => { console.error(e); process.exit(1); });
