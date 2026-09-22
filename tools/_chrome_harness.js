// Isolated Chrome for SpawnScene browser gates.
//
// Gates must NOT run in the developer's own browser. Doing so leaks tabs into it, competes for
// the GPU they are using, requires them to have launched with --remote-debugging-port, and a
// publish mid-run white-pages whatever they had open. This launches a separate Chrome with its
// own profile and debug port, and kills exactly the PID it started (never by image name - that
// would take down the daily browser).
//
// Uses the INSTALLED Chrome, not a bundled Chromium: bundled Chromium falls back to SOFTWARE
// WebGPU, which makes every timing and some behaviour unrepresentative.

const http = require('http');
const os = require('os');
const path = require('path');
const fs = require('fs');
const { spawn } = require('child_process');

const CHROME = process.env.SPAWNSCENE_CHROME
  || 'C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe';
// Agent split on one machine: Trip 9223, Tuvok 9224 (set SPAWNSCENE_CDP_PORT). Sharing a
// port used to silently attach and then one agent would kill or starve the other's GPU run.
const PORT = parseInt(process.env.SPAWNSCENE_CDP_PORT || '9223', 10);
// Opt-in only. Default is launch-our-own: attaching to "whatever is on 9223" is how Trip and
// Tuvok ended up driving the same Chrome (and blaming each other when it died).
const ATTACH = process.env.SPAWNSCENE_CDP_ATTACH === '1';

function probe(port) {
  return new Promise((res) => {
    const req = http.get(`http://127.0.0.1:${port}/json/version`, r => {
      let d = ''; r.on('data', c => d += c);
      r.on('end', () => { try { res(JSON.parse(d)); } catch { res(null); } });
    });
    req.on('error', () => res(null));
    req.setTimeout(700, () => { req.destroy(); res(null); });
  });
}

/**
 * Ensure a debuggable Chrome is available.
 * Returns { port, endpoint, close() }. close() is a no-op if we attached to an existing one:
 * never kill a browser this process did not start.
 */
async function ensureChrome({ headless = false } = {}) {
  const existing = await probe(PORT);
  if (existing) {
    if (ATTACH) {
      console.log(`[chrome] attaching to existing instance on ${PORT} (SPAWNSCENE_CDP_ATTACH=1)`);
      return { port: PORT, endpoint: existing.webSocketDebuggerUrl, close: async () => {} };
    }
    throw new Error(
      `Chrome already answers on CDP port ${PORT}. Another agent (or a leftover harness) owns ` +
      `it - do NOT attach by default. Free that port, or set SPAWNSCENE_CDP_PORT to a free one ` +
      `(Trip=9223, Tuvok=9224). SPAWNSCENE_CDP_ATTACH=1 only when you mean to share.`);
  }

  const profile = path.join(os.tmpdir(), `spawnscene-harness-${PORT}`);
  fs.mkdirSync(profile, { recursive: true });

  const args = [
    `--remote-debugging-port=${PORT}`,
    `--user-data-dir=${profile}`,
    '--no-first-run',
    '--no-default-browser-check',
    '--disable-background-timer-throttling',
    '--disable-renderer-backgrounding',
    '--disable-backgrounding-occluded-windows',
    // Keep OPFS across runs in this profile so a generated scene can be reused.
    'about:blank',
  ];
  if (headless) args.unshift('--headless=new');

  console.log(`[chrome] launching ${path.basename(CHROME)} on ${PORT} (profile ${profile})`);
  const child = spawn(CHROME, args, { detached: false, stdio: 'ignore' });

  const deadline = Date.now() + 30000;
  while (Date.now() < deadline) {
    const v = await probe(PORT);
    if (v) {
      console.log(`[chrome] ready: ${v.Browser}`);
      return {
        port: PORT,
        endpoint: v.webSocketDebuggerUrl,
        close: async () => {
          try {
            // Ask politely first so the profile is flushed, then make sure it is gone.
            await new Promise(res => http.get(`http://127.0.0.1:${PORT}/json/close`, r => { r.resume(); r.on('end', res); }).on('error', () => res()));
          } catch { /* ignore */ }
          try { process.kill(child.pid); } catch { /* already gone */ }
          console.log('[chrome] closed the instance this run launched');
        },
      };
    }
    await new Promise(r => setTimeout(r, 400));
  }
  try { process.kill(child.pid); } catch { /* ignore */ }
  throw new Error(`Chrome did not expose CDP on ${PORT} within 30s`);
}

module.exports = { ensureChrome, probe, PORT };
