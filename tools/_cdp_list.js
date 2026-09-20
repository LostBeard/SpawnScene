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
  for (const p of pages) {
    console.log(JSON.stringify({ type: p.type, title: p.title, url: p.url }));
  }

  for (const p of pages.filter((x) => x.type === 'page')) {
    try {
      const ws = new WebSocket(p.webSocketDebuggerUrl);
      await new Promise((r, j) => {
        ws.on('open', r);
        ws.on('error', j);
      });
      let id = 1;
      const pend = new Map();
      ws.on('message', (raw) => {
        const m = JSON.parse(raw);
        if (m.id && pend.has(m.id)) pend.get(m.id)(m);
      });
      const send = (method, params) =>
        new Promise((res) => {
          const i = id++;
          pend.set(i, res);
          ws.send(JSON.stringify({ id: i, method, params }));
        });
      await send('Runtime.enable');
      const ev = await send('Runtime.evaluate', {
        expression:
          '({href:location.href,title:document.title,buttons:document.querySelectorAll("button").length,texts:[...document.querySelectorAll("button")].map(b=>b.textContent.trim()).filter(Boolean).slice(0,20)})',
        returnByValue: true,
      });
      console.log('EVAL', p.id.slice(0, 8), JSON.stringify(ev.result?.result?.value));
      ws.close();
    } catch (e) {
      console.log('fail', p.id.slice(0, 8), e.message);
    }
  }
})();
