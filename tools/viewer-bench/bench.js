// Shared by the third-party viewer pages: the poses and options from the URL, a requestAnimationFrame frame counter, and
// the harness handshake (tools/_cdp_viewer_bench.js): per pose, set the camera, warm up, count frames for the measure
// window, log "[Bench] <viewer> POSE k READY fps F", then wait for the harness to take its screenshot.
export function options() {
    const q = new URLSearchParams(location.search);
    return {
        url: q.get('url'), fov: +(q.get('fov') || 50), posesUrl: q.get('poses'),
        warm: +(q.get('warm') || 4000), measure: +(q.get('measure') || 8000),
    };
}
let frames = 0;
(function tick() { frames++; requestAnimationFrame(tick); })();
const sleep = ms => new Promise(r => setTimeout(r, ms));
const GLIDE = 1500;
const lerp = (a, b, t) => a.map((v, i) => v + (b[i] - v) * t);
async function glide(a, b, setPose) {
    const t0 = performance.now();
    for (;;) {
        const t = Math.min(1, (performance.now() - t0) / GLIDE);
        const s = t * t * (3 - 2 * t);
        setPose({ pos: lerp(a.pos, b.pos, s), target: lerp(a.target, b.target, s), up: lerp(a.up, b.up, s) });
        if (t >= 1) return;
        await new Promise(r => requestAnimationFrame(r));
    }
}
export async function run(name, setPose) {
    const o = options();
    const poses = (await (await fetch(o.posesUrl)).json()).ply;
    let prev = null;
    for (let k = 0; k < poses.length; k++) {
        // Glide from the previous pose over GLIDE ms, as a person's camera moves: a one-frame jump left Spark drawing
        // with a depth order from the previous camera (mirrored far wall on top) at some poses.
        if (prev) await glide(prev, poses[k], setPose);
        setPose(poses[k]); prev = poses[k];
        await sleep(o.warm);
        const f0 = frames, t0 = performance.now();
        await sleep(o.measure);
        const fps = (frames - f0) * 1000 / (performance.now() - t0);
        console.log(`[Bench] ${name} POSE ${k} READY fps ${fps.toFixed(1)}`);
        await new Promise(r => { window.__benchNext = r; });
    }
    console.log(`[Bench] ${name} DONE`);
}
