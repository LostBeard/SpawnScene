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
  let pages = await getJson('http://127.0.0.1:9222/json/list');
  let page = pages.find((p) => p.type === 'page' && /8080|studio/i.test(p.url || ''));
  // Prefer a live page target; if chrome-error, still use it to navigate
  page = pages.find((p) => p.type === 'page' && (p.url || '').includes('8080/studio'))
    || pages.find((p) => p.type === 'page' && (p.url || '').includes('8080'))
    || pages.find((p) => p.type === 'page' && /chrome-error|studio/i.test(p.url || ''));
  if (!page) {
    // Open new tab via browser target
    const version = await getJson('http://127.0.0.1:9222/json/version');
    console.log('opening new tab via browser');
    const bws = new WebSocket(version.webSocketDebuggerUrl);
    await new Promise((r) => bws.on('open', r));
    bws.send(JSON.stringify({ id: 1, method: 'Target.createTarget', params: { url: 'http://127.0.0.1:8080/studio' } }));
    await new Promise((r) => setTimeout(r, 2000));
    bws.close();
    pages = await getJson('http://127.0.0.1:9222/json/list');
    page = pages.find((p) => p.type === 'page' && (p.url || '').includes('8080'));
  }
  console.log('PAGE', page && page.url, page && page.title);
  if (!page) process.exit(2);

  const ws = new WebSocket(page.webSocketDebuggerUrl);
  let id = 1;
  const pending = new Map();
  const logs = [];

  function send(method, params = {}) {
    return new Promise((resolve, reject) => {
      const i = id++;
      const t = setTimeout(() => {
        if (pending.has(i)) {
          pending.delete(i);
          reject(new Error('timeout ' + method));
        }
      }, 180000);
      pending.set(i, {
        resolve: (v) => {
          clearTimeout(t);
          resolve(v);
        },
        reject,
      });
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
      const args = (msg.params.args || []).map((a) => a.value ?? a.description ?? '').join(' ');
      if (/MultiView|DepthGPU|Temple|World-space|Error|splat|Depth\]/i.test(args)) {
        logs.push(args);
        console.log('CON:', args);
      }
    }
    if (msg.method === 'Page.loadEventFired') {
      console.log('LOAD');
    }
  });

  await new Promise((r) => ws.on('open', r));
  await send('Runtime.enable');
  await send('Page.enable');
  await send('Network.setCacheDisabled', { cacheDisabled: true }).catch(() => {});

  console.log('Navigate...');
  await send('Page.navigate', { url: 'http://127.0.0.1:8080/studio' });

  // Wait for Blazor / TempleRing button
  let ready = false;
  for (let i = 0; i < 60; i++) {
    await new Promise((r) => setTimeout(r, 1000));
    const ev = await send('Runtime.evaluate', {
      expression: `({href:location.href, buttons:[...document.querySelectorAll('button')].map(b=>b.textContent.trim()).filter(Boolean).slice(0,40)})`,
      returnByValue: true,
    });
    const v = ev.result?.result?.value;
    console.log('poll', i, v && v.href, v && (v.buttons || []).filter((b) => /Temple/i.test(b)));
    if (v && /studio/i.test(v.href || '') && (v.buttons || []).some((b) => /TempleRing/i.test(b))) {
      ready = true;
      break;
    }
  }
  if (!ready) {
    console.log('NOT READY');
    process.exit(3);
  }

  const click = await send('Runtime.evaluate', {
    expression: `(()=>{const b=[...document.querySelectorAll('button')].find(x=>/TempleRing/i.test(x.textContent||'')); if(!b) return 'missing'; b.click(); return 'clicked';})()`,
    returnByValue: true,
  });
  console.log('CLICK', click.result?.result?.value);

  const deadline = Date.now() + 300000;
  while (Date.now() < deadline) {
    await new Promise((r) => setTimeout(r, 2000));
    if (logs.some((l) => /No views produced|GT complete:|TempleRing GT scene:|TempleRing error/i.test(l))) break;
  }

  console.log('---SUMMARY---');
  logs.filter((l) => /World-space|View \d|Total:|Error|GT complete|TempleRing GT|mvs_fuse|affine|pose=gt/i.test(l)).forEach((l) => console.log(l));
  const mvs = logs.some((l) => /mvs_fuse kept=/i.test(l));
  const pose = logs.some((l) => /pose=gt\+mvs/i.test(l));
  const nonzero = logs.some((l) => /GT complete: [1-9].*pose=gt\+mvs|TempleRing GT scene: [1-9]/.test(l));
  console.log(JSON.stringify({ mvs, pose, nonzero, logCount: logs.length }));
  ws.close();
  process.exit(nonzero && mvs && pose ? 0 : 1);
})().catch((e) => {
  console.error(e);
  process.exit(1);
});
