const http = require("http");
const fs = require("fs");
const path = require("path");
const root = path.resolve(process.argv[2]);

// Mount each dataset's images from wherever they actually live, read from its manifest.
//
// A large dataset is not copied into wwwroot: Deep Blending's drjohnson is 168 MB of JPEG,
// and anything under wwwroot is recopied on every publish - this repo has already had a
// publish balloon to 13 GB from output landing inside the project. So the manifest that
// tools/colmap_to_dataset.py writes records the source directory, and this reads it.
//
// Config comes from the manifest rather than an env var on purpose. The first version took
// MOUNTS="/datasets/X/images=C:/path", and Git Bash's POSIX path translation rewrote BOTH
// halves into nonsense before node ever saw them. A file the script wrote itself cannot be
// mangled by a shell.
const mounts = [];
for (const dsRoot of [path.join(root, 'datasets'),
                      path.resolve(__dirname, '..', 'SpawnScene', 'wwwroot', 'datasets')]) {
  let names = [];
  try { names = fs.readdirSync(dsRoot); } catch { continue; }
  for (const name of names) {
    const mf = path.join(dsRoot, name, 'manifest.json');
    try {
      const m = JSON.parse(fs.readFileSync(mf, 'utf8'));
      if (!m.source || !m.imageDir) continue;
      const dir = path.join(m.source, m.imageDir);
      if (!fs.existsSync(dir)) continue;
      const prefix = `/datasets/${name}/${m.imageDir}`;
      if (mounts.some(x => x.prefix === prefix)) continue;
      mounts.push({ prefix, dir });
    } catch { /* not a mounted dataset */ }
  }
}
for (const m of mounts) console.log(`mounted ${m.prefix} -> ${m.dir}`);
const port = +process.argv[3] || 8080;
const mime = { ".html":"text/html",".js":"text/javascript",".mjs":"text/javascript",".css":"text/css",".wasm":"application/wasm",".json":"application/json",".png":"image/png",".svg":"image/svg+xml",".woff2":"font/woff2",".dll":"application/octet-stream",".dat":"application/octet-stream",".pdb":"application/octet-stream",".map":"application/json",".mp4":"video/mp4",".webm":"video/webm",".mov":"video/quicktime",".jpg":"image/jpeg",".jpeg":"image/jpeg" };
http.createServer((req,res)=>{
  let url = decodeURIComponent((req.url||"/").split("?")[0]);
  if (url === "/") url = "/index.html";
  let file = path.join(root, url);
  for (const m of mounts) {
    if (url === m.prefix || url.startsWith(m.prefix + '/')) {
      file = path.join(m.dir, url.slice(m.prefix.length));
      break;
    }
  }
  const trySend = (f, code) => {
    fs.readFile(f, (err, data) => {
      if (err) {
        if (code !== 200) { res.writeHead(404); return res.end("404"); }
        return trySend(path.join(root, "index.html"), 200);
      }
      const ext = path.extname(f).toLowerCase();
      res.writeHead(200, {"Content-Type": mime[ext]||"application/octet-stream", "Cache-Control":"no-cache", "Access-Control-Allow-Origin":"*"});
      res.end(data);
    });
  };
  fs.stat(file, (err, st) => {
    if (!err && st.isDirectory()) file = path.join(file, "index.html");
    if (err) {
      // SPA fallback only for extension-less routes (/studio). Never for .png/.wasm/etc.
      const ext = path.extname(url).toLowerCase();
      if (ext) {
        res.writeHead(404, { "Content-Type": "text/plain" });
        return res.end("404");
      }
      return trySend(path.join(root, "index.html"), 200);
    }
    trySend(file, 200);
  });
}).listen(port, "127.0.0.1", () => console.log("SPA listening", port, root));
