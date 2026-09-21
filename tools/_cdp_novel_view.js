// Novel-view fidelity capture.
//
// TempleRing reconstructs from 4 of its 16 photos. This drives the app to the ground-truth pose
// of each of the OTHER views and captures what it renders, so the result can be scored against
// the real photograph. A pose the reconstruction never saw is exactly what looking around
// produces, so this measures navigation correctness and render correctness in one shot.
//
//   node _cdp_novel_view.js [RUN_TAG] [maxViews]
//
// Writes _shots/novelview/<RUN_TAG>/<view>.png plus poses.json, then score with:
//   python tools/score_novel_view.py <RUN_TAG> [<BASELINE_TAG>]
//
// TAB DISCIPLINE: exactly ONE tab for the whole run, reused by navigation, closed in a finally
// so a crash or Ctrl-C cannot leak it. An earlier version opened a tab per view and only closed
// them at the very end, so a failed run left every one of them resident - each a live WebGPU
// context on somebody else machine. Resource use on someone else machine is correctness.
//
// Per-run output directories are also deliberate: a gate that writes fixed filenames destroys
// the baseline by running, which already cost this project one before/after comparison.

const http = require('http');
const fs = require('fs');
const path = require('path');
const WebSocket = require('ws');
const { ensureChrome } = require('./_chrome_harness');

// Repo root, derived from this file so the tool works from any clone.
const ROOT = path.resolve(__dirname, '..').replace(/\\/g, '/');
const DATASET = path.join(ROOT, 'SpawnScene/wwwroot/datasets/TempleRing');
const RUN_TAG = process.argv[2] || new Date().toISOString().replace(/[:.]/g, '-');
const MAX_VIEWS = parseInt(process.argv[3] || '6', 10);
// Extra query args, e.g. ONLYVIEW=0 to unproject a single view (diagnostic).
let EXTRA = '';
if (process.env.ONLYVIEW) EXTRA += `&onlyview=${process.env.ONLYVIEW}`;
if (process.env.GLOBALSCALE) EXTRA += `&globalscale=${process.env.GLOBALSCALE}`;
// TRAIN=<iters> runs photometric optimisation against the posed photographs BEFORE the first
// pose is parked, on the same page load, so the captured frames are of the trained scene.
if (process.env.TRAIN) EXTRA += `&train=${process.env.TRAIN}`;
// UPRIGHT=1 stands each source photograph up before depth inference. TempleRing's images are
// all a quarter turn off level, which is out of distribution for a monocular depth model.
// Uses its own saved scene, so it never reuses the other orientation's geometry.
if (process.env.UPRIGHT) EXTRA += `&upright=${process.env.UPRIGHT}`;
const OUT = path.join(ROOT, '_shots/novelview', RUN_TAG);

// GT render size. Must match the dataset images or the comparison resamples.
const VW = 640, VH = 480;

let CDP = 9223; // set once the harness Chrome is up
const cdp = (p) => `http://127.0.0.1:${CDP}${p}`;

function get(u) {
  return new Promise((res, rej) =>
    http.get(u, (r) => { let d = ''; r.on('data', c => d += c); r.on('end', () => res(JSON.parse(d))); })
      .on('error', rej));
}
function closeTab(id) {
  return new Promise(res =>
    http.get(cdp('/json/close/' + id), r => { r.resume(); r.on('end', res); })
      .on('error', () => res()));
}

// The app picks training views by farthest-point sampling over the images ON DISK and logs the
// indices; reproduce that ordering so we know which views are held out.
const imagesOnDisk = () =>
  fs.readdirSync(DATASET).filter(f => /^templeR\d+\.png$/i.test(f)).sort();

(async () => {
  fs.mkdirSync(OUT, { recursive: true });
  const all = imagesOnDisk();
  console.log(`dataset: ${all.length} images on disk`);

  // Own browser, own profile. Never the developer's.
  const chrome = await ensureChrome();
  CDP = chrome.port;

  // ── Open the single tab, BLANK ──
  // Blank first: applying the viewport override to an already-booting app resizes the canvas
  // mid-initialization, tears down WebGPU under the running depth inference ("A valid external
  // Instance reference no longer exists") and the depth pass returns 0 views.
  const before = new Set((await get(cdp('/json/list'))).map(t => t.id));
  const ver = await get(cdp('/json/version'));
  const bws = new WebSocket(ver.webSocketDebuggerUrl);
  await new Promise(r => bws.on('open', r));
  bws.send(JSON.stringify({ id: 1, method: 'Target.createTarget', params: { url: 'about:blank' } }));
  await new Promise(r => setTimeout(r, 800));
  bws.close();

  const tab = (await get(cdp('/json/list')))
    .find(t => t.type === 'page' && !before.has(t.id));
  if (!tab) throw new Error('could not open a tab');

  let closed = false;
  const cleanup = async () => {
    if (closed) return;
    closed = true;
    await closeTab(tab.id);
    console.log('closed the tab this run opened');
  };
  process.on('SIGINT', async () => { await cleanup(); process.exit(130); });

  const results = [];
  let trainingNames = [];

  try {
    const ws = new WebSocket(tab.webSocketDebuggerUrl);
    await new Promise(r => ws.on('open', r));
    let id = 1;
    const pend = new Map();
    let logs = [];
    ws.on('message', (raw) => {
      const m = JSON.parse(raw.toString());
      if (m.id && pend.has(m.id)) pend.get(m.id)(m);
      if (m.method === 'Runtime.consoleAPICalled') {
        const s = (m.params.args || []).map(a => a.value ?? a.description ?? '').join(' ');
        logs.push(s);
        // Wide on purpose: pipeline diagnostics ([MultiView-GT] keepRatio, per-view counts,
        // confidence retained) were being filtered out, so I was reading a metric without
        // being able to confirm the code that produced it had run.
        if (/NovelView|MultiView|DepthGPU|Train|Trainer|farthest picks|FAIL|Error/i.test(s)) console.log('  CON', s.slice(0, 200));
      }
    });
    const send = (method, params = {}) => new Promise((res, rej) => {
      const i = id++;
      const to = setTimeout(() => rej(new Error('timeout ' + method)), 900000);
      pend.set(i, v => { clearTimeout(to); res(v); });
      ws.send(JSON.stringify({ id: i, method, params }));
    });

    await send('Runtime.enable');
    await send('Page.enable');
    await send('Emulation.setDeviceMetricsOverride',
      { width: VW, height: VH, deviceScaleFactor: 1, mobile: false });

    // Wait for THIS view READY marker, not any READY. One page serves every pose, so the
    // console buffer holds the previous view marker; a bare /READY/ match would return
    // instantly and screenshot the wrong pose. Assert the view name.
    async function waitReady(view, timeoutMs, since) {
      // Plain substring, NOT a regex. Inside a template literal the escapes collapse, so
      // "\[NovelView\]" becomes a character CLASS and "\b" becomes a literal backspace that can
      // never match - every view would have silently timed out.
      const marker = `[NovelView] READY view=${view}`;
      const deadline = Date.now() + timeoutMs;
      while (Date.now() < deadline) {
        if (logs.slice(since).some(l => l.includes(marker))) return;
        const fail = logs.slice(since).find(l => /\[NovelView\] FAIL/.test(l));
        if (fail) throw new Error(fail.slice(0, 300));
        await new Promise(r => setTimeout(r, 400));
      }
      throw new Error(`timed out waiting for READY ${view}`);
    }

    async function capture(view) {
      const shot = await send('Page.captureScreenshot', { format: 'png' });
      fs.writeFileSync(path.join(OUT, view), Buffer.from(shot.result.data, 'base64'));
      results.push(view);
    }

    // A hash change does NOT reload the page, so the generated scene stays resident and each
    // extra view costs only a camera move. Re-navigating instead made the app re-decide
    // "is there a saved scene?" from an unsettled OPFS listing, and views randomly regenerated.
    async function gotoView(view, timeoutMs) {
      const since = logs.length;
      await send('Runtime.evaluate', { expression: `location.hash = 'view=${view}'` });
      await waitReady(view, timeoutMs, since);
    }

    // ── Pass 1: one page load. Generates (or reuses) the scene and parks the first pose. ──
    const firstView = all[all.length - 1];
    console.log(`
=== boot + generate (${firstView}) ===`);
    await send('Page.navigate',
      { url: `http://127.0.0.1:8080/studio?autotest=novel-view&view=${firstView}${EXTRA}&cb=${Date.now()}` });
    await waitReady(firstView, 900000, 0);

    const picks = logs.find(l => /farthest picks/.test(l));
    if (picks) {
      const m = picks.match(/\[([\d,\s]+)\]/);
      if (m) trainingNames = m[1].split(',').map(x => all[parseInt(x.trim(), 10)]).filter(Boolean);
    }
    const canvas = await send('Runtime.evaluate', {
      expression: `(()=>{const c=document.querySelector('canvas');return c?c.width+'x'+c.height:'none';})()`,
      returnByValue: true,
    });
    console.log('  canvas:', canvas.result.result.value);
    await capture(firstView);

    console.log(`
training views: ${trainingNames.join(', ') || '(unknown, scoring all)'}`);
    const heldOut = all.filter(f => !trainingNames.includes(f));

    // Capture TRAINING views too. They are the upper bound and the single most diagnostic
    // measurement available: if the reconstruction cannot reproduce a photo it was BUILT from,
    // the fault is fundamental and novel-view quality is a red herring. Scored separately.
    const order = [
      ...trainingNames,
      ...heldOut.filter(f => f !== firstView).slice(0, Math.max(0, MAX_VIEWS - 1)),
    ];
    console.log(`queue: ${order.length} view(s) (${trainingNames.length} training)`);

    for (const view of order) {
      if (results.includes(view)) continue;
      process.stdout.write(`  ${view} ... `);
      try {
        await gotoView(view, 120000);
        await capture(view);
        console.log('ok');
      } catch (e) {
        console.log(`SKIP (${e.message.slice(0, 90)})`);
      }
    }

    await send('Runtime.evaluate', { expression: `location.hash = 'done'` }).catch(() => {});

    fs.writeFileSync(path.join(OUT, 'poses.json'), JSON.stringify({
      runTag: RUN_TAG, captured: results, trainingViews: trainingNames, heldOut, viewport: [VW, VH],
    }, null, 2));
    ws.close();
  } finally {
    await cleanup();
    await chrome.close();
  }

  console.log(`\ncaptured ${results.length} view(s) -> ${OUT}`);
  console.log(`score with: python tools/score_novel_view.py ${RUN_TAG}`);
  process.exit(results.length > 0 ? 0 : 1);
})().catch(async (e) => { console.error(e.message || e); process.exit(1); });
