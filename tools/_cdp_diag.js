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
  const page = pages.find((p) => p.type === 'page');
  const ws = new WebSocket(page.webSocketDebuggerUrl);
  await new Promise((r) => ws.on('open', r));
  let id = 1;
  const pend = new Map();
  ws.on('message', (raw) => {
    const m = JSON.parse(raw);
    if (m.id && pend.has(m.id)) pend.get(m.id)(m);
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

  // Read error page body
  const body = await send('Runtime.evaluate', {
    expression: 'document.body ? document.body.innerText.slice(0,1500) : "no body"',
    returnByValue: true,
  });
  console.log('BODY:', body.result?.result?.value);

  // Try fetch from page context
  const fet = await send('Runtime.evaluate', {
    expression: `(async()=>{ try { const r=await fetch('http://127.0.0.1:8080/'); return 'ok '+r.status+' '+ (await r.text()).length; } catch(e){ return 'err '+e.message; } })()`,
    awaitPromise: true,
    returnByValue: true,
  });
  console.log('FETCH:', fet.result?.result?.value);

  // Navigate again and capture network failure
  const fails = [];
  ws.on('message', (raw) => {
    const m = JSON.parse(raw.toString());
    if (m.method === 'Network.loadingFailed') {
      fails.push(m.params);
      console.log('FAIL', JSON.stringify(m.params));
    }
    if (m.method === 'Network.responseReceived') {
      console.log('RESP', m.params.response.status, m.params.response.url);
    }
  });

  await send('Page.navigate', { url: 'http://127.0.0.1:8080/studio' });
  await new Promise((r) => setTimeout(r, 5000));
  console.log('fails', fails.length);

  const after = await send('Runtime.evaluate', {
    expression: 'location.href + " :: " + (document.body&&document.body.innerText||"").slice(0,500)',
    returnByValue: true,
  });
  console.log('AFTER:', after.result?.result?.value);
  ws.close();
})();
