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
const { ensureChrome } = require('./_chrome_harness');

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
const EXTRA = process.env.EXTRA || '';
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
        const free = s.match(/\[Dataset\] READY-FOR-CAPTURE free-(\w+)/);
        if (free) { pendingFree.push(free[1]); }
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
    const url = `http://127.0.0.1:8080/studio?autotest=dataset&name=${NAME}&train=${TRAIN}${GEOM}${MAXDIM}${POSES}${PATCHES}${ANCHORS}${NVIEWS}${BUDGET}${MAXSCALE}${POSLR}${HELDEVERY}${SKIPZEROGRAD}${GTPOSES}${DENSIFY}${EXTRA}&cb=${Date.now()}`;
    console.log(`\n=== ${NAME}, ${TRAIN} iters ===\n${url}\n`);
    await send('Page.navigate', { url });

    const deadline = Date.now() + 55 * 60 * 1000;
    let shot = false;
    while (Date.now() < deadline && !done && !failed) {
      await new Promise(r => setTimeout(r, 500));
      // Capture the finished scene, so the reconstruction can be LOOKED at and not only scored.
      while (pendingFree.length) {
        const name = pendingFree.shift();
        const png = await send('Page.captureScreenshot', { format: 'png' });
        const dir = path.join(__dirname, '..', '_shots', 'dataset');
        require('fs').mkdirSync(dir, { recursive: true });
        const tag = process.env.RUN_TAG ? `${NAME}__${process.env.RUN_TAG}__free-${name}.png`
                                        : `${NAME}__free-${name}.png`;
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
    if (failed) { console.log('\nFAILED'); process.exitCode = 1; }
    else if (!done) { console.log('\nTIMED OUT'); process.exitCode = 1; }
    else console.log('\nOK');
  } finally {
    await cleanup();
    await chrome.close();
  }
})().catch(e => { console.error(e); process.exit(1); });
