const http = require('http');
const WebSocket = require('ws');
const fs = require('fs');
const REPO = require('path').resolve(__dirname, '..').replace(/\\/g, '/');

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
  let page = pages.find((p) => p.type === 'page' && (p.url || '').includes('8080/studio'));
  if (!page) {
    const ver = await get('http://127.0.0.1:9222/json/version');
    const bws = new WebSocket(ver.webSocketDebuggerUrl);
    await new Promise((r) => bws.on('open', r));
    bws.send(JSON.stringify({ id: 1, method: 'Target.createTarget', params: { url: 'http://127.0.0.1:8080/studio' } }));
    await new Promise((r) => setTimeout(r, 2500));
    bws.close();
    const pages2 = await get('http://127.0.0.1:9222/json/list');
    page = pages2.find((p) => p.type === 'page' && (p.url || '').includes('8080'));
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
      if (/MultiView|DepthGPU|Temple|World-space|Error|Studio|splat|Depth\]|pick|Consistency/i.test(t)) {
        logs.push(t);
        console.log('CON', t);
      }
    }
  });
  const send = (method, params = {}) =>
    new Promise((res, rej) => {
      const i = id++;
      const to = setTimeout(() => rej(new Error('timeout ' + method)), 300000);
      pend.set(i, (v) => {
        clearTimeout(to);
        res(v);
      });
      ws.send(JSON.stringify({ id: i, method, params }));
    });

  await send('Runtime.enable');
  await send('Page.enable');
  await send('Network.setCacheDisabled', { cacheDisabled: true }).catch(() => {});

  async function click(x, y) {
    await send('Input.dispatchMouseEvent', { type: 'mouseMoved', x, y });
    await send('Input.dispatchMouseEvent', { type: 'mousePressed', x, y, button: 'left', clickCount: 1 });
    await send('Input.dispatchMouseEvent', { type: 'mouseReleased', x, y, button: 'left', clickCount: 1 });
  }
  // Fixed filenames only: every run used to overwrite the previous run's shots, so the gate
  // could never show a before/after and a baseline was destroyed by the act of measuring.
  // RUN_TAG keeps a permanent copy per run under _shots/.
  const RUN_TAG = process.env.RUN_TAG || new Date().toISOString().replace(/[:.]/g, '-');
  fs.mkdirSync(REPO + '/_shots', { recursive: true });
  async function shot(name) {
    const r = await send('Page.captureScreenshot', { format: 'png' });
    const buf = Buffer.from(r.result.data, 'base64');
    fs.writeFileSync(require('path').resolve(__dirname, '..').replace(/\\/g, '/') + '/' + name, buf);
    fs.writeFileSync(REPO + '/_shots/' + RUN_TAG + '_' + name, buf);
    console.log('shot', name, '(also _shots/' + RUN_TAG + '_' + name + ')');
  }

  console.log('navigate');
  await send('Page.navigate', { url: 'http://127.0.0.1:8080/studio?cb=' + Date.now() });
  await new Promise((r) => setTimeout(r, 14000));
  logs.length = 0; // drop prior-page console noise
  await shot('_t1.png');

  // New Project → opens detail (after republish). Button at (170,160)
  console.log('New Project');
  await click(170, 160);
  await new Promise((r) => setTimeout(r, 2500));
  await shot('_t2.png');

  // If still on browser (old build), click Open around card
  if (!logs.some((l) => /Created project|Project Detail|TempleRing/i.test(l))) {
    console.log('try Open');
    await click(120, 250);
    await new Promise((r) => setTimeout(r, 1500));
    await shot('_t2b.png');
  }

  // TempleRing button ~ (70+70, 40+204+13) = (140, 257) for empty project
  console.log('TempleRing clicks');
  for (const y of [250, 257, 265, 280, 300, 320, 340, 360]) {
    await click(140, y);
    await new Promise((r) => setTimeout(r, 600));
    if (logs.some((l) => /TempleRing:/i.test(l))) break;
  }
  await shot('_t3.png');

  const deadline = Date.now() + 360000;
  while (Date.now() < deadline) {
    await new Promise((r) => setTimeout(r, 2000));
    if (logs.some((l) => /No views produced|GT complete:|TempleRing GT scene:|TempleRing error|Umeyama poor|monocular fallback/i.test(l))) {
      if (logs.some((l) => /No views produced|GT complete:|TempleRing GT scene:|TempleRing error/i.test(l))) break;
    }
  }
  // Settle + frontal screenshot before orbit
  await new Promise((r) => setTimeout(r, 1500));
  await shot('_temple_front.png');
  // gentle orbit + screenshot
  const cx = 480, cy = 470;
  await send('Input.dispatchMouseEvent', { type: 'mousePressed', x: cx, y: cy, button: 'left', clickCount: 1 });
  for (let i = 1; i <= 15; i++)
    await send('Input.dispatchMouseEvent', { type: 'mouseMoved', x: cx + i * 6, y: cy - i, button: 'left' });
  await send('Input.dispatchMouseEvent', { type: 'mouseReleased', x: cx + 90, y: cy - 15, button: 'left', clickCount: 1 });
  await new Promise((r) => setTimeout(r, 1500));
  await shot('_temple_aligned.png');

  console.log('---SUMMARY---');
  logs.forEach((l) => console.log(l));
  const joint = logs.some((l) => /pose=gt\+mvs/i.test(l));
  const cull = logs.some((l) => /Sphere cull:|Culling background/i.test(l));
  const fuse = logs.some((l) => /mvs_fuse kept=/i.test(l));
  const affine = logs.some((l) => /affine-per-view|affine a=/i.test(l));
  const views4 = logs.filter((l) => /View \d+ affine/i.test(l)).length >= 4
    || logs.some((l) => /farthest picks:.*?,.*,.*,/i.test(l));
  const ok = logs.some((l) => /TempleRing GT scene:\s*[1-9]|GT complete:\s*[1-9].*pose=gt\+mvs/.test(l));
  const fail = logs.some((l) => /FAIL (affine_probe|mvs_fuse|joint_depth)/i.test(l));
  const voxel = logs.some((l) => /mvs_voxel|dense-gpu|cleanedPx=/i.test(l));
  const mono = logs.some((l) => /monocular fallback|pose=gt \(monocular/i.test(l));
  const dense = (() => {
    const m = logs.map((l) => l.match(/GT complete:\s*([\d,]+)/i)).find((x) => x);
    if (!m) return false;
    const n = parseInt(m[1].replace(/,/g, ''), 10);
    return n >= 50000;
  })();
  console.log(JSON.stringify({ joint, cull, fuse, affine, views4, voxel, ok, fail, mono, dense, n: logs.length }));
  ws.close();
  process.exit(ok && joint && fuse && !fail && !mono && dense ? 0 : 1);
})().catch((e) => {
  console.error(e);
  process.exit(1);
});
