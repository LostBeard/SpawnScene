// Drive the whole pipeline on an UNPOSED capture.
//
// TempleRing ships a calibration file; nothing else in datasets/ does. This runs the ordinary
// path - load images, recover poses, estimate depth, optimise - on a dataset that has no poses,
// which is what the product is actually for.
//
//   node _cdp_dataset.js [NAME] [TRAIN_ITERS]
//   GEOM=1 also optimises position/scale/rotation.  MAXDIM=720 caps the training resolution.
//
// Reports rather than gates: the question is whether the recovered poses are good enough to
// optimise against, and a pass/fail threshold here would be a guess dressed up as a gate.

const http = require('http');
const path = require('path');
const WebSocket = require('ws');
const { ensureChrome, APP } = require('./_chrome_harness');

const NAME = process.argv[2] || 'Bathroom';
const TRAIN = parseInt(process.argv[3] || '1600', 10);
const GEOM = process.env.GEOM ? `&geom=${process.env.GEOM}` : '';
const MAXDIM = process.env.MAXDIM ? `&maxdim=${process.env.MAXDIM}` : '';
// POSES=dav3 keeps depth and cameras in one frame by skipping SfM entirely.
const POSES = process.env.POSES ? `&poses=${process.env.POSES}` : '';
// PATCHES=N sets the depth ViT patch budget. 1369 (=37*37) matches the old 518 square.
const PATCHES = process.env.PATCHES ? `&patches=${process.env.PATCHES}` : '';
// ANCHORS=N views shared by every chunked pass; N=N views per joint forward.
const ANCHORS = process.env.ANCHORS ? `&anchors=${process.env.ANCHORS}` : '';
const NVIEWS = process.env.N ? `&n=${process.env.N}` : '';
// EXTRA passes anything else straight through, so a new knob does not need a new harness.
// Quote EXTRA when setting it in cmd.exe: set "EXTRA=&init=points" — a bare & starts a new command.
const EXTRA = process.env.EXTRA || '';
// INIT=points uses the dataset's sparse SfM cloud (reference create_from_pcd).
const INIT = process.env.INIT ? `&init=${process.env.INIT}` : '';
// BUDGET=N splats for the initialisation.
const BUDGET = process.env.BUDGET ? `&budget=${process.env.BUDGET}` : '';
// MAXSCALE=N caps a splat at N * scene diagonal; POSLR=N scales the position learning rate.
const MAXSCALE = process.env.MAXSCALE ? `&maxscale=${process.env.MAXSCALE}` : '';
const POSLR = process.env.POSLR ? `&poslr=${process.env.POSLR}` : '';
// HELDEVERY=N evaluates held-out PSNR every N cycles, so the curve is visible, not just its ends.
const HELDEVERY = process.env.HELDEVERY ? `&heldevery=${process.env.HELDEVERY}` : '';
// SKIPZEROGRAD=1 guards the colour/opacity Adam step against zero-gradient splats.
const SKIPZEROGRAD = process.env.SKIPZEROGRAD ? `&skipzerograd=${process.env.SKIPZEROGRAD}` : '';
// GTPOSES=1 uses a dataset's own COLMAP poses instead of recovering them.
const GTPOSES = process.env.GTPOSES ? `&gtposes=${process.env.GTPOSES}` : '';
// DENSIFY=N runs adaptive density control every N cycles.
const DENSIFY = process.env.DENSIFY ? `&densify=${process.env.DENSIFY}` : '';
// DENSIFYGRAD=X is the densify |grad| bar in peak-|dCentre| PIXEL units (default 1.5e-6).
const DENSIFYGRAD = process.env.DENSIFYGRAD ? `&densifygrad=${process.env.DENSIFYGRAD}` : '';
// DENSIFYFRAC=X keeps only the top fraction of above-threshold candidates (Brush: 0.2).
const DENSIFYFRAC = process.env.DENSIFYFRAC ? `&densifyfrac=${process.env.DENSIFYFRAC}` : '';
// MAXDENSIFY=N caps splat count during densify (default 450k in app).
const MAXDENSIFY = process.env.MAXDENSIFY ? `&maxdensify=${process.env.MAXDENSIFY}` : '';
// DENSIFYUNTIL=N stops densify after iteration N (Kerbl: 15000 absolute).
const DENSIFYUNTIL = process.env.DENSIFYUNTIL ? `&densifyuntil=${process.env.DENSIFYUNTIL}` : '';
// OPACITYRESET=N caps every opacity every N cycles and unlocks the size prunes.
const OPACITYRESET = process.env.OPACITYRESET ? `&opacityreset=${process.env.OPACITYRESET}` : '';
// FITONE=N trains against view N alone: a capacity ceiling for the rasteriser.
const FITONE = process.env.FITONE ? `&fitone=${process.env.FITONE}` : '';

let CDP = 9223;
const cdp = (p) => `http://127.0.0.1:${CDP}${p}`;
const get = (u) => new Promise((res, rej) =>
  http.get(u, r => { let d = ''; r.on('data', c => d += c); r.on('end', () => res(JSON.parse(d))); })
    .on('error', rej));
const closeTab = (id) => new Promise(res =>
  http.get(cdp('/json/close/' + id), r => { r.resume(); r.on('end', res); }).on('error', () => res()));

(async () => {
  const chrome = await ensureChrome();
  CDP = chrome.port;

  const before = new Set((await get(cdp('/json/list'))).map(t => t.id));
  const ver = await get(cdp('/json/version'));
  const bws = new WebSocket(ver.webSocketDebuggerUrl);
  await new Promise(r => bws.on('open', r));
  bws.send(JSON.stringify({ id: 1, method: 'Target.createTarget', params: { url: 'about:blank' } }));
  await new Promise(r => setTimeout(r, 800));
  bws.close();

  const tab = (await get(cdp('/json/list'))).find(t => t.type === 'page' && !before.has(t.id));
  if (!tab) throw new Error('could not open a tab');

  let closed = false;
  const cleanup = async () => { if (!closed) { closed = true; await closeTab(tab.id); } };
  process.on('SIGINT', async () => { await cleanup(); process.exit(130); });

  try {
    const ws = new WebSocket(tab.webSocketDebuggerUrl);
    await new Promise(r => ws.on('open', r));
    let id = 1;
    const pend = new Map();
    let done = false, failed = null, ready = false;
    const pendingFree = [];
    const viewMap = {};
    const pendingTrainer = [];   // view-<k>: the trainer's own render of that pose, on #trainerdump
    const pendingPhoto = [];     // view-<k>: the photograph when it is a video frame (in memory), on #photodump
    ws.on('message', raw => {
      const m = JSON.parse(raw.toString());
      if (m.id && pend.has(m.id)) pend.get(m.id)(m);
      if (m.method === 'Runtime.consoleAPICalled') {
        const s = (m.params.args || []).map(a => a.value ?? a.description ?? '').join(' ');
        // Print EVERYTHING. This used to be an allowlist of prefixes - Dataset, Train,
        // MultiView, SfM, Studio, Depth - and a new [Densify] prefix matched none of them, so
        // nine density-control steps produced no output and were reported as "it did nothing".
        // The diagnosis cost two full runs and was wrong.
        //
        // An allowlist of message prefixes drops exactly the output you added because something
        // was unclear. Output goes to a redirected file anyway, and grep is free; GPU minutes
        // are not. Same lesson as the pose-source whitelist, third time in one day.
        console.log(s.slice(0, 400));
        // free-<name>: a pose nobody stood at. view-<sup|held>-<i> <photo url> <WxH>: parked at a real
        // photo's pose and intrinsics, for a side-by-side with that photo (tools/compose_views.py).
        const free = s.match(/\[Dataset\] READY-FOR-CAPTURE free-(\w+)/);
        const view = s.match(/\[Dataset\] READY-FOR-CAPTURE view-(\w+-\d+) (\S+) (\d+x\d+)/);
        const trainerRender = s.match(/\[Dataset\] TRAINER-RENDER (\S+)/);
        if (trainerRender) pendingTrainer.push(trainerRender[1]);
        const photoDump = s.match(/\[Dataset\] PHOTO-DUMP (\S+)/);
        if (photoDump) pendingPhoto.push(photoDump[1]);
        if (free) { pendingFree.push(free[1]); }
        else if (view) { pendingFree.push('view-' + view[1]); viewMap[view[1]] = { photo: view[2], size: view[3] }; }
        else if (/\[Dataset\] READY-FOR-CAPTURE/.test(s)) ready = true;
        if (/\[Dataset\] DONE/.test(s)) done = true;
        if (/\[Dataset\] FAIL/.test(s)) failed = s;
      }
    });
    const send = (method, params = {}) => new Promise((res, rej) => {
      const i = id++;
      const to = setTimeout(() => rej(new Error('timeout ' + method)), 3600000);
      pend.set(i, v => { clearTimeout(to); res(v); });
      ws.send(JSON.stringify({ id: i, method, params }));
    });

    await send('Runtime.enable');
    await send('Page.enable');
    // VIEWPORT=WxH: render at the dataset's photo size, dpr 1, so view-* captures are pixel-aligned with
    // the photographs they are compared to (as _cdp_novel_view.js does for TempleRing).
    if (process.env.VIEWPORT) {
      const [vw, vh] = process.env.VIEWPORT.split('x').map(Number);
      await send('Emulation.setDeviceMetricsOverride', { width: vw, height: vh, deviceScaleFactor: 1, mobile: false });
      console.log(`[harness] viewport ${vw}x${vh} @1x`);
    }
    const url = `${APP}/studio?autotest=dataset&name=${NAME}&train=${TRAIN}${GEOM}${MAXDIM}${POSES}${PATCHES}${ANCHORS}${NVIEWS}${BUDGET}${MAXSCALE}${POSLR}${HELDEVERY}${SKIPZEROGRAD}${GTPOSES}${DENSIFY}${DENSIFYGRAD}${DENSIFYFRAC}${MAXDENSIFY}${DENSIFYUNTIL}${OPACITYRESET}${FITONE}${INIT}${EXTRA}&cb=${Date.now()}`;
    console.log(`\n=== ${NAME}, ${TRAIN} iters ===\n${url}\n`);
    await send('Page.navigate', { url });

    const deadline = Date.now() + 55 * 60 * 1000;
    let shot = false;
    while (Date.now() < deadline && !done && !failed) {
      await new Promise(r => setTimeout(r, 500));
      // The trainer's render of a view, saved next to the viewer's capture of the same pose.
      while (pendingTrainer.length) {
        const name = pendingTrainer.shift();
        const r = await send('Runtime.evaluate', {
          expression: "document.getElementById('trainerdump').toDataURL('image/png')", returnByValue: true });
        const dataUrl = r.result && r.result.result && r.result.result.value;
        if (typeof dataUrl === 'string' && dataUrl.startsWith('data:image/png;base64,')) {
          const dir = path.join(__dirname, '..', '_shots', 'dataset');
          require('fs').mkdirSync(dir, { recursive: true });
          const tag = `${NAME}__${process.env.RUN_TAG || 'untagged'}__${name}-trainer.png`;
          require('fs').writeFileSync(path.join(dir, tag), Buffer.from(dataUrl.slice(22), 'base64'));
          console.log('captured ' + tag);
        } else console.log('trainer render ' + name + ': no canvas data');
      }
      while (pendingPhoto.length) {
        const name = pendingPhoto.shift();
        const r = await send('Runtime.evaluate', {
          expression: "document.getElementById('photodump').toDataURL('image/png')", returnByValue: true });
        const dataUrl = r.result && r.result.result && r.result.result.value;
        if (typeof dataUrl === 'string' && dataUrl.startsWith('data:image/png;base64,')) {
          const dir = path.join(__dirname, '..', '_shots', 'dataset');
          require('fs').mkdirSync(dir, { recursive: true });
          const tag = `${NAME}__${process.env.RUN_TAG || 'untagged'}__${name}-photo.png`;
          require('fs').writeFileSync(path.join(dir, tag), Buffer.from(dataUrl.slice(22), 'base64'));
          console.log('captured ' + tag);
        } else console.log('photo dump ' + name + ': no canvas data');
      }
      // Capture the finished scene, so the reconstruction can be LOOKED at and not only scored.
      while (pendingFree.length) {
        const name = pendingFree.shift();
        const png = await send('Page.captureScreenshot', { format: 'png' });
        const dir = path.join(__dirname, '..', '_shots', 'dataset');
        require('fs').mkdirSync(dir, { recursive: true });
        const kind = name.startsWith('view-') ? name : `free-${name}`;
        const tag = process.env.RUN_TAG ? `${NAME}__${process.env.RUN_TAG}__${kind}.png`
                                        : `${NAME}__${kind}.png`;
        require('fs').writeFileSync(path.join(dir, tag), Buffer.from(png.result.data, 'base64'));
        console.log('captured ' + tag);
      }
      if (ready && !shot) {
        shot = true;
        const png = await send('Page.captureScreenshot', { format: 'png' });
        const dir = path.join(__dirname, '..', '_shots', 'dataset');
        require('fs').mkdirSync(dir, { recursive: true });
        const bytes = Buffer.from(png.result.data, 'base64');
        // Also write a TAGGED copy. A gate whose only output has a fixed name destroys its own
        // baseline by running, and the question about a render is always "what is different",
        // which needs two pictures. RUN_TAG=<name> to label this one.
        const out = path.join(dir, `${NAME}.png`);
        require('fs').writeFileSync(out, bytes);
        console.log('captured ' + out);
        if (process.env.RUN_TAG) {
          const tagged = path.join(dir, `${NAME}__${process.env.RUN_TAG}.png`);
          require('fs').writeFileSync(tagged, bytes);
          console.log('captured ' + tagged);
        }
      }
    }
    if (Object.keys(viewMap).length) {
      const dir = path.join(__dirname, '..', '_shots', 'dataset');
      const side = path.join(dir, `${NAME}__${process.env.RUN_TAG || 'untagged'}__views.json`);
      require('fs').writeFileSync(side, JSON.stringify({ app: APP, views: viewMap }, null, 2));
      console.log('wrote ' + side);
    }
    if (failed) { console.log('\nFAILED'); process.exitCode = 1; }
    else if (!done) { console.log('\nTIMED OUT'); process.exitCode = 1; }
    else console.log('\nOK');
  } finally {
    await cleanup();
    await chrome.close();
  }
})().catch(e => { console.error(e); process.exit(1); });
