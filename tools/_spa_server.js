const http = require("http");
const fs = require("fs");
const path = require("path");
const root = path.resolve(process.argv[2]);
const port = +process.argv[3] || 8080;
const mime = { ".html":"text/html",".js":"text/javascript",".mjs":"text/javascript",".css":"text/css",".wasm":"application/wasm",".json":"application/json",".png":"image/png",".svg":"image/svg+xml",".woff2":"font/woff2",".dll":"application/octet-stream",".dat":"application/octet-stream",".pdb":"application/octet-stream",".map":"application/json" };
http.createServer((req,res)=>{
  let url = decodeURIComponent((req.url||"/").split("?")[0]);
  if (url === "/") url = "/index.html";
  let file = path.join(root, url);
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
