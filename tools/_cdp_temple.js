const { APP } = require('./_chrome_harness');
const http = require('http');
const WebSocket = require('ws');

function getJson(url) {
  return new Promise((resolve, reject) => {
    http.get(url, (res) => {
      let d = '';
      res.on('data', (c) => (d += c));
      res.on('end', () => {
        try { resolve(JSON.parse(d)); } catch (e) { reject(e); }
      });
    }).on('error', reject);
  });
}

(async () => {
  const pages = (await getJson('http://127.0.0.1:9222/json/list'))
    .filter((p) => p.type === 'page' && p.url && p.url.includes('8080'));
  let page = pages.find((p) => p.url.includes('/studio')) || pages[0];
  if (!page) {
    console.log('NO_STUDIO');
    process.exit(2);
  }
  console.log('PAGE', page.url, page.title);

  const ws = new WebSocket(page.webSocketDebuggerUrl);
  let id = 1;
  const pending = new Map();
  const logs = [];

  function send(method, params = {}) {
    return new Promise((resolve, reject) => {
      const i = id++;
      pending.set(i, { resolve, reject });
      ws.send(JSON.stringify({ id: i, method, params }));
      setTimeout(() => {
        if (pending.has(i)) {
          pending.delete(i);
          reject(new Error('timeout ' + method));
        }
      }, 120000);
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
      if (/MultiView|DepthGPU|Temple|World-space|Error|splat/i.test(args)) {
        logs.push(args);
        console.log('CON:', args);
      }
    }
  });

  await new Promise((r) => ws.on('open', r));
  await send('Runtime.enable');
  await send('Page.enable');

  console.log('Reloading...');
  await send('Page.reload', { ignoreCache: true });
  await new Promise((r) => setTimeout(r, 8000));

  // Navigate to studio if needed
  const urlEval = await send('Runtime.evaluate', {
    expression: 'location.href',
    returnByValue: true,
  });
  console.log('URL', urlEval.result?.result?.value);
  if (!(urlEval.result?.result?.value || '').includes('/studio')) {
    await send('Page.navigate', { url: APP + '/studio' });
    await new Promise((r) => setTimeout(r, 8000));
  }

  // Click TempleRing button by text
  const click = await send('Runtime.evaluate', {
    expression: `(async function(){
      const wait = (ms)=>new Promise(r=>setTimeout(r,ms));
      for (let i=0;i<40;i++){
        const buttons=[...document.querySelectorAll('button')];
        const b=buttons.find(x=>/TempleRing/i.test(x.textContent||''));
        if(b){ b.click(); return 'clicked:'+b.textContent.trim(); }
        await wait(500);
      }
      return 'NOT_FOUND buttons='+[...document.querySelectorAll('button')].map(b=>b.textContent.trim()).slice(0,30).join('|');
    })()`,
    awaitPromise: true,
    returnByValue: true,
  });
  console.log('CLICK', JSON.stringify(click.result?.result?.value || click));

  // Wait for completion (up to ~4 min for depth+splats)
  const deadline = Date.now() + 240000;
  while (Date.now() < deadline) {
    await new Promise((r) => setTimeout(r, 2000));
    const done = logs.some((l) =>
      /No views produced|GT complete:|TempleRing GT scene:|Total:.*pose=gt|TempleRing error/i.test(l)
    );
    if (done) break;
  }

  console.log('---SUMMARY---');
  const interesting = logs.filter((l) => /World-space|View \\d|Total:|Error|GT complete|TempleRing GT|mvs_fuse|affine/i.test(l));
  interesting.forEach((l) => console.log(l));
  const mvs = logs.some((l) => /mvs_fuse kept=/i.test(l));
  const pose = logs.some((l) => /pose=gt\+mvs/i.test(l));
  const zero = logs.some((l) => /World-space.*: 0 splats|No views produced|FAIL mvs_fuse/i.test(l));
  const ok = logs.some((l) => /GT complete: [1-9].*pose=gt\+mvs|TempleRing GT scene: [1-9]/i.test(l));
  console.log(JSON.stringify({ mvs, pose, zero, ok, logCount: logs.length }));
  ws.close();
  process.exit(ok && mvs && pose ? 0 : 1);
})().catch((e) => {
  console.error(e);
  process.exit(1);
});
