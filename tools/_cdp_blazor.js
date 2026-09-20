const http = require('http');
const WebSocket = require('ws');

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
  const pages = await get('http://127.0.0.1:9222/json/list');
  const page = pages.find((p) => p.type === 'page' && (p.url || '').includes('8080'));
  const ws = new WebSocket(page.webSocketDebuggerUrl);
  await new Promise((r) => ws.on('open', r));
  let id = 1;
  const pend = new Map();
  const logs = [];
  const fails = [];
  ws.on('message', (raw) => {
    const m = JSON.parse(raw.toString());
    if (m.id && pend.has(m.id)) pend.get(m.id)(m);
    if (m.method === 'Runtime.consoleAPICalled') {
      const t = (m.params.args || []).map((a) => a.value ?? a.description ?? '').join(' ');
      logs.push(t);
      console.log('CON', t.slice(0, 300));
    }
    if (m.method === 'Runtime.exceptionThrown') {
      console.log('EXC', m.params.exceptionDetails?.text, m.params.exceptionDetails?.exception?.description);
    }
    if (m.method === 'Network.loadingFailed') {
      fails.push(m.params);
      console.log('NETFAIL', m.params.errorText, m.params.type);
    }
  });
  const send = (method, params = {}) =>
    new Promise((res) => {
      const i = id++;
      pend.set(i, res);
      ws.send(JSON.stringify({ id: i, method, params }));
    });

  await send('Runtime.enable');
  await send('Network.enable');
  await send('Page.enable');
  await send('Page.reload', { ignoreCache: true });
  await new Promise((r) => setTimeout(r, 15000));

  const info = await send('Runtime.evaluate', {
    expression: `({
      href: location.href,
      title: document.title,
      buttons: document.querySelectorAll('button').length,
      app: !!document.querySelector('#app'),
      scripts: [...document.scripts].map(s=>s.src).slice(0,10),
      body: (document.body&&document.body.innerText||'').slice(0,800)
    })`,
    returnByValue: true,
  });
  console.log('INFO', JSON.stringify(info.result?.result?.value, null, 2));
  console.log('fails', fails.length, fails.slice(0, 10));
  ws.close();
})();
