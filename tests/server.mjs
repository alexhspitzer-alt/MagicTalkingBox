import { createServer } from 'node:http';
import { readFile } from 'node:fs/promises';
import path from 'node:path';
const types = { '.html': 'text/html', '.js': 'application/javascript', '.css': 'text/css' };
createServer(async (request, response) => {
  const url = new URL(request.url, 'http://localhost');
  if (!url.pathname.startsWith('/MagicTalkingBox/')) { response.writeHead(404).end(); return; }
  const relative = decodeURIComponent(url.pathname.slice('/MagicTalkingBox/'.length)) || 'index.html';
  const filename = path.resolve('dist', relative);
  if (!filename.startsWith(path.resolve('dist') + path.sep)) { response.writeHead(403).end(); return; }
  try {
    const data = await readFile(filename);
    response.writeHead(200, { 'Content-Type': types[path.extname(filename)] || 'application/octet-stream', 'Cache-Control': 'no-store' }).end(data);
  } catch { response.writeHead(404).end(); }
}).listen(4173, '127.0.0.1');
