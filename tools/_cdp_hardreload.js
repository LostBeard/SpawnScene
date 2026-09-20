const http = require('http'); const WebSocket = require('ws');
const getJson = (u) => new Promise((res, rej) => http.get(u, (r) => { let d=''; r.on('data',c=>d+=c); r.on('end',()=>res(JSON.parse(d))); }).on('error', rej));
(async () => {
  const pages = (await getJson('http://127.0.0.1:9222/json/list')).filter(p => p.type === 'page' && p.url && p.url.includes('8080'));
  const page = pages.find(p => p.url.includes('/studio')) || pages[0];
  const ws = new WebSocket(page.webSocketDebuggerUrl); let id = 1;
  const send = (method, params={}) => new Promise(r => { ws.send(JSON.stringify({id:id++, method, params})); setTimeout(r, 200); });
  await new Promise(r => ws.on('open', r));
  await send('Network.enable');
  await send('Network.setCacheDisabled', { cacheDisabled: true });
  await send('Page.enable');
  await send('Page.reload', { ignoreCache: true });
  console.log('hard reload issued, cache disabled');
  await new Promise(r => setTimeout(r, 12000));
  ws.close();
})();
