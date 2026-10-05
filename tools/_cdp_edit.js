// Scene editing in the desktop viewer, driven with real mouse events (CDP Input.dispatchMouseEvent): loads the Room
// sample, opens Edit, selects with a dragged rectangle, then Delete -> Undo -> Keep only, capturing each step and
// printing the [Edit] log lines (selected counts, undo depth).
//
//   SPAWNSCENE_CDP_PORT=9228 SPAWNSCENE_APP_PORT=8102 node tools/_cdp_edit.js [outPrefix]
const http = require('http');
const fs = require('fs');
const WebSocket = require('ws');
const { ensureChrome, APP } = require('./_chrome_harness');

const [prefix = '_shots/edit'] = process.argv.slice(2);
const get = u => new Promise((res, rej) =>
  http.get(u, r => { let d = ''; r.on('data', c => d += c); r.on('end', () => res(JSON.parse(d))); }).on('error', rej));
const sleep = ms => new Promise(r => setTimeout(r, ms));

(async () => {
  const chrome = await ensureChrome();
  const cdp = p => `http://127.0.0.1:${chrome.port}${p}`;
  const before = new Set((await get(cdp('/json/list'))).map(t => t.id));
  const ver = await get(cdp('/json/version'));
  const bws = new WebSocket(ver.webSocketDebuggerUrl);
  await new Promise(r => bws.on('open', r));
  bws.send(JSON.stringify({ id: 1, method: 'Target.createTarget', params: { url: 'about:blank' } }));
  await sleep(800);
  bws.close();
  const tab = (await get(cdp('/json/list'))).find(t => t.type === 'page' && !before.has(t.id));
  if (!tab) throw new Error('no tab');

  const ws = new WebSocket(tab.webSocketDebuggerUrl);
  await new Promise(r => ws.on('open', r));
  let id = 1; const pend = new Map(); let passed = false;
  ws.on('message', raw => {
    const m = JSON.parse(raw.toString());
    if (m.id && pend.has(m.id)) pend.get(m.id)(m);
    // Downloads: say what Chrome does with them (a 737 MB export reported DONE and never landed).
    if (m.method === 'Page.downloadWillBegin') console.log('DL begin ' + m.params.suggestedFilename);
    if (m.method === 'Page.downloadProgress' && m.params.state !== 'inProgress') console.log('DL ' + m.params.state + ' ' + (m.params.receivedBytes || 0) + ' bytes');
    if (m.method === 'Runtime.exceptionThrown') console.log('EXC ' + JSON.stringify(m.params.exceptionDetails).slice(0, 300));
    if (m.method === 'Runtime.consoleAPICalled') {
      const t = (m.params.args || []).map(a => a.value ?? a.description ?? '').join(' ');
      if ((process.env.SPAWNSCENE_EDIT_WAIT ? new RegExp(process.env.SPAWNSCENE_EDIT_WAIT) : /\[Autotest\] PASS/).test(t)) passed = true;
      if (process.env.SPAWNSCENE_EDIT_PRINT && new RegExp(process.env.SPAWNSCENE_EDIT_PRINT).test(t)) console.log('CON ' + t.slice(0, 400));
      else if (/error/.test(m.params.type) || /\[Autotest\]|\[Edit\]|\[Import\]|\[Dataset\] (FAIL|DONE)|\[Studio\] (scene saved|Loaded|scene .*SH)|GPU ERROR/.test(t)) console.log('CON ' + t.slice(0, 300));
    }
  });
  const send = (method, params = {}) => new Promise(res => { const i = id++; pend.set(i, res); ws.send(JSON.stringify({ id: i, method, params })); });
  const shot = async name => {
    const png = await send('Page.captureScreenshot', { format: 'png' });
    fs.writeFileSync(`${prefix}_${name}.png`, Buffer.from(png.result.data, 'base64'));
    console.log(`captured ${prefix}_${name}.png`);
  };
  const touch = (type, pts) => send('Input.dispatchTouchEvent', { type, touchPoints: pts.map(([x, y], i) => ({ x, y, id: i })) });
  // A gesture as a sequence of finger sets, one frame apart.
  const gesture = async frames => {
    await touch('touchStart', frames[0]);
    for (const f of frames.slice(1)) { await sleep(33); await touch('touchMove', f); }
    await sleep(33);
    await touch('touchEnd', []);
  };
  const mouse = (type, x, y) => send('Input.dispatchMouseEvent', { type, x, y, button: 'left', buttons: type === 'mouseReleased' ? 0 : 1, clickCount: 1 });
  const click = async (x, y) => { await send('Input.dispatchMouseEvent', { type: 'mouseMoved', x, y }); await sleep(60); await mouse('mousePressed', x, y); await sleep(80); await mouse('mouseReleased', x, y); await sleep(300); };
  const drag = async (x0, y0, x1, y1) => {
    await send('Input.dispatchMouseEvent', { type: 'mouseMoved', x: x0, y: y0 }); await sleep(60);
    await mouse('mousePressed', x0, y0);
    for (let k = 1; k <= 10; k++) { await sleep(33); await send('Input.dispatchMouseEvent', { type: 'mouseMoved', x: x0 + (x1 - x0) * k / 10, y: y0 + (y1 - y0) * k / 10, buttons: 1 }); }
    await sleep(60); await mouse('mouseReleased', x1, y1); await sleep(600);
  };
  try {
    await send('Page.enable');
    await send('Runtime.enable');
    await send('Emulation.setDeviceMetricsOverride', { width: 1600, height: 1000, deviceScaleFactor: 1, mobile: false });
    // SPAWNSCENE_EDIT_DOWNLOADS: where downloads land, for every flow (e.g. QUERY=export=latest with FLOW=none).
    if (process.env.SPAWNSCENE_EDIT_DOWNLOADS) {
      const dl = require('path').resolve(process.env.SPAWNSCENE_EDIT_DOWNLOADS);
      fs.mkdirSync(dl, { recursive: true });
      await send('Page.setDownloadBehavior', { behavior: 'allow', downloadPath: dl });
    }
    // SPAWNSCENE_EDIT_QUERY: another start (e.g. the project autotest, which trains and saves a scene), with
    // SPAWNSCENE_EDIT_WAIT the console line that says it is done.
    await send('Page.navigate', { url: `${APP}/studio?${process.env.SPAWNSCENE_EDIT_QUERY || 'autotest=generate-room&render=stochastic'}` });
    // SPAWNSCENE_EDIT_MINUTES: how long SPAWNSCENE_EDIT_QUERY may run (a 30K project training is well past an hour).
    const deadline = Date.now() + Number(process.env.SPAWNSCENE_EDIT_MINUTES || 60) * 60 * 1000;
    while (!passed && Date.now() < deadline) await sleep(500);
    if (!passed) throw new Error('the Room sample never passed');
    await sleep(1500);
    if (process.env.SPAWNSCENE_EDIT_FLOW === 'none') {   // just run SPAWNSCENE_EDIT_QUERY to its WAIT line
      // ...and, with SPAWNSCENE_EDIT_DOWNLOADS, until no download is still in flight (.crdownload) - a 400 MB scene
      // export outlives the WAIT line by many seconds, and closing the tab cancels it.
      if (process.env.SPAWNSCENE_EDIT_DOWNLOADS) {
        const dl = require('path').resolve(process.env.SPAWNSCENE_EDIT_DOWNLOADS);
        const t0 = Date.now();
        while (Date.now() - t0 < 10 * 60 * 1000) {
          const f = fs.readdirSync(dl);
          if (f.length > 0 && !f.some(n => n.endsWith('.crdownload'))) break;
          await sleep(2000);
        }
        console.log('downloads: ' + fs.readdirSync(dl).join(', '));
      }
      // SPAWNSCENE_EDIT_SHOT=1: a screenshot of what the viewer ends on (e.g. is the scene upright).
      if (process.env.SPAWNSCENE_EDIT_SHOT) { await sleep(5000); await shot('final'); }
      return;
    }
    if (process.env.SPAWNSCENE_EDIT_FLOW === 'export') {
      // After the project autotest: open its scene, Edit -> Export file, the download saved to SPAWNSCENE_EDIT_DOWNLOADS.
      const dir = require('path').resolve(process.env.SPAWNSCENE_EDIT_DOWNLOADS || '_shots/downloads');
      fs.mkdirSync(dir, { recursive: true });
      await send('Page.setDownloadBehavior', { behavior: 'allow', downloadPath: dir });
      await click(253, 264); await sleep(10000);          // Open (first scene card)
      await shot('e0_scene');
      await click(1288, 28); await sleep(600);            // Edit (a project scene's viewer: no Depth button)
      await click(90, 517); await sleep(60000);           // Export file (the 11th toolbar button)
      console.log('downloads: ' + fs.readdirSync(dir).join(', '));
      return;
    }
    if (process.env.SPAWNSCENE_EDIT_FLOW === 'trained') {
      // After the project autotest (trained scene saved, project page shown): open it, copy/paste with its SH bands,
      // save as a new scene, open that. The viewer of a project scene has no Depth button: Edit sits left of AR.
      await shot('t0_project');
      await click(253, 264); await sleep(8000);          // Open (first scene card)
      await shot('t1_trained');
      await click(1288, 28); await sleep(600);            // Edit
      await click(90, 97);                                // Select
      await drag(300, 150, 700, 800);     // right of the toolbar (a drag that starts on it is a toolbar click)
      await click(90, 223); await sleep(2500);            // Copy
      await click(90, 307); await sleep(4000);            // Paste
      await shot('t2_pasted');
      await click(90, 475); await sleep(10000);           // Save as new scene
      await click(80, 28); await sleep(3000);             // back to the project
      await shot('t3_project');
      await click(795, 264); await sleep(8000);           // Open the second card (the edited copy)
      await shot('t4_reopened');
      return;
    }
    // Top bar: Edit sits left of AR (canvas 1600 wide, Send to Headset hidden on loopback). Toolbar: left edge,
    // buttons 42 px apart from y 97: Select, Delete, Keep only, Copy, Cut, Paste, Undo, Clear selection, Save.
    const Y = { select: 97, del: 139, keep: 181, copy: 223, cut: 265, paste: 307, insert: 349, undo: 391, clear: 433, save: 475 };
    await click(1208, 28); await sleep(500);
    if (process.env.SPAWNSCENE_EDIT_FLOW === 'insert') {
      // SPAWNSCENE_EDIT_FLOW=insert: Insert scene -> the newest other saved scene (this profile keeps earlier runs'
      // projects), then back off with touch pinch-ins to see both side by side.
      await click(90, Y.insert); await sleep(1500);   // Insert scene >
      await shot('i0_list');
      await click(338, 119); await sleep(5000);  // the first scene in the list
      await shot('i1_inserted');
      await send('Emulation.setTouchEmulationEnabled', { enabled: true, maxTouchPoints: 5 });
      const touch = (type, pts) => send('Input.dispatchTouchEvent', { type, touchPoints: pts.map(([x, y], i) => ({ x, y, id: i })) });
      for (let n = 0; n < 3; n++) {
        await touch('touchStart', [[600, 500], [1000, 500]]);
        for (let k = 1; k <= 10; k++) { await sleep(33); await touch('touchMove', [[600 + 17.5 * k, 500], [1000 - 17.5 * k, 500]]); }
        await sleep(33); await touch('touchEnd', []);
      }
      await sleep(2500);
      await shot('i2_backed_off');
      return;
    }
    if (process.env.SPAWNSCENE_EDIT_FLOW === 'copy') {
      // SPAWNSCENE_EDIT_FLOW=copy: select the red pillow, Copy, Paste (lands to its right), Undo.
      await click(90, Y.select);
      await drag(720, 570, 910, 735);
      await shot('c0_selected');
      await click(90, Y.copy); await sleep(1500);
      await click(90, Y.paste); await sleep(2500);
      await shot('c1_pasted');
      // Move the pasted copy (selected after the paste): Up twice. Move buttons, 2 columns under the status line.
      await click(56, 603); await sleep(1200);
      await click(56, 603); await sleep(1500);
      await shot('c1b_moved_up');
      await click(90, Y.undo); await sleep(1500);
      await shot('c2_undone');
      return;
    }
    await click(90, Y.select);                   // Select
    await drag(600, 520, 1000, 800);             // over the sofa and table
    await shot('0_selected');
    await click(90, Y.del); await sleep(1500);   // Delete
    await shot('1_deleted');
    await click(90, Y.undo); await sleep(1500);  // Undo
    await shot('2_undone');
    await drag(600, 520, 1000, 800);             // select again (still in Select mode)
    await click(90, Y.keep); await sleep(1500);  // Keep only
    await shot('3_kept');
    await click(90, Y.save); await sleep(4000);  // Save as new scene
    await shot('4_saved');
    await click(80, 28); await sleep(2500);      // back to the project page
    await shot('5_project');
    // SPAWNSCENE_EDIT_OPEN="x,y": then click there (a scene card) and capture the reopened scene.
    if (process.env.SPAWNSCENE_EDIT_OPEN) {
      const [ox, oy] = process.env.SPAWNSCENE_EDIT_OPEN.split(',').map(Number);
      await click(ox, oy); await sleep(6000);
      await shot('6_reopened');
    }
  } finally {
    try { ws.send(JSON.stringify({ id: id++, method: 'Page.close', params: {} })); } catch { }
    await sleep(300);
    try { ws.close(); } catch { }
    if (chrome.close) await chrome.close();
  }
})().then(() => process.exit(0), e => { console.error(e); process.exit(1); });
