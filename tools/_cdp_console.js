const http = require('http');
const WebSocket = require('ws');

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
  console.log('pages', pages.map((p) => ({ title: p.title, url: p.url })));
  const page = pages.find((p) => p.url.includes('/studio')) || pages[0];
  if (!page) {
    console.log('no studio page');
    return;
  }

  const ws = new WebSocket(page.webSocketDebuggerUrl);
  let id = 1;
  const pending = new Map();
  const consoleLines = [];

  function send(method, params = {}) {
    return new Promise((resolve, reject) => {
      const i = id++;
      pending.set(i, { resolve, reject });
      ws.send(JSON.stringify({ id: i, method, params }));
    });
  }

  ws.on('message', (raw) => {
    const msg = JSON.parse(raw.toString());
    if (msg.id && pending.has(msg.id)) {
      const { resolve } = pending.get(msg.id);
      pending.delete(msg.id);
      resolve(msg);
      return;
    }
    if (msg.method === 'Runtime.consoleAPICalled') {
      const args = (msg.params.args || []).map((a) => a.value ?? a.description ?? JSON.stringify(a)).join(' ');
      consoleLines.push(args);
    }
  });

  await new Promise((r) => ws.on('open', r));
  await send('Runtime.enable');
  await send('Log.enable');
  await send('Console.enable').catch(() => ({}));

  const r = await send('Runtime.evaluate', {
    expression: `(function(){
      const t = document.body ? document.body.innerText : '';
      const lines = t.split(/\\n/).filter(l => /splat|MultiView|Error|World|Depth|Temple|status/i.test(l));
      return { totalLen: t.length, lines: lines.slice(-60) };
    })()`,
    returnByValue: true,
  });
  console.log('PAGE_STATUS', JSON.stringify(r.result?.result?.value || r, null, 2));

  // Also try to pull performance logs / any stored console if Blazor exposed it
  const r2 = await send('Runtime.evaluate', {
    expression: `(function(){
      try {
        if (window.__spawnLogs) return window.__spawnLogs.slice(-80);
      } catch(e) {}
      return null;
    })()`,
    returnByValue: true,
  });
  console.log('SPAWN_LOGS', JSON.stringify(r2.result?.result?.value, null, 2));

  // Wait briefly for any live console
  await new Promise((r) => setTimeout(r, 500));
  console.log('LIVE_CONSOLE_COUNT', consoleLines.length);
  consoleLines.slice(-40).forEach((l) => console.log('CON:', l));

  ws.close();
})().catch((e) => {
  console.error(e);
  process.exit(1);
});
