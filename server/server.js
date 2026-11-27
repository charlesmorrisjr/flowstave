'use strict';
/*
 * Flowstave server
 * - Reads every MIDI keyboard connected to this computer (via RtMidi).
 * - Stamps each note with this computer's high-resolution clock the moment it arrives.
 * - Sends notes to every connected browser (iPad, laptop...) over a WebSocket.
 * - Answers clock-sync pings so browsers can convert the PC's time stamps to their own clock.
 * - Serves the app from ./public.
 *
 * Settings (environment variables):
 *   PORT        port to listen on (default 3000)
 *   HOST        address to listen on (default 0.0.0.0 = all network interfaces)
 *   MIDI_FILTER only open MIDI inputs whose name contains this text (case-insensitive)
 * HTTPS: put key.pem and cert.pem in ./certs (for example made with mkcert) and the
 * server switches to HTTPS + secure WebSockets automatically.
 */
const http = require('http');
const https = require('https');
const fs = require('fs');
const os = require('os');
const path = require('path');
const { performance } = require('perf_hooks');
const { WebSocketServer, WebSocket } = require('ws');
const ai = require('./ai/generate');

const PORT = Number(process.env.PORT) || 3000;
const HOST = process.env.HOST || '0.0.0.0';
const MIDI_FILTER = (process.env.MIDI_FILTER || '').toLowerCase();
const PUBLIC_DIR = path.join(__dirname, 'public');
const CERT_DIR = path.join(__dirname, 'certs');
const MODELS_DIR = path.join(__dirname, 'models');
const POLL_MS = 2000;

const now = () => performance.now(); // monotonic milliseconds, sub-millisecond precision
const log = (...a) => console.log(new Date().toLocaleTimeString(), ...a);

/* ---------------- web server ---------------- */
const MIME = {
  '.html': 'text/html; charset=utf-8', '.js': 'text/javascript; charset=utf-8', '.css': 'text/css; charset=utf-8',
  '.json': 'application/json', '.png': 'image/png', '.svg': 'image/svg+xml', '.ico': 'image/x-icon', '.webmanifest': 'application/manifest+json'
};
const who = req => (req.socket.remoteAddress || '').replace(/^::ffff:/, '');
function describeBrowser(ua = '') {
  const ios = ua.match(/OS (\d+)[_.](\d+)(?:[_.](\d+))? like Mac OS X/);
  const ver = ua.match(/Version\/([\d.]+)/);
  if (ios) return `${/iPad/.test(ua) ? 'iPad' : /iPhone/.test(ua) ? 'iPhone' : 'iOS device'}, iOS ${ios[1]}.${ios[2]}${ios[3] ? '.' + ios[3] : ''}${ver ? ', Safari ' + ver[1] : ''}`;
  if (/Macintosh/.test(ua) && ver && !/Chrome|Chromium|Edg/.test(ua)) return `Safari ${ver[1]} (Mac, or an iPad requesting the desktop site)`;
  const m = ua.match(/(Edg|Chrome|Firefox)\/([\d.]+)/);
  return m ? `${m[1] === 'Edg' ? 'Edge' : m[1]} ${m[2]}` : ua.slice(0, 120);
}
function serveStatic(req, res) {
  let urlPath;
  try { urlPath = decodeURIComponent(new URL(req.url, 'http://x').pathname); } catch { res.writeHead(400); return res.end(); }
  // Errors reported by the page itself (shown here so problems on the iPad are visible on the PC).
  if (req.method === 'POST' && urlPath === '/client-log') {
    let body = '';
    req.on('data', c => { if (body.length < 4000) body += c; });
    req.on('end', () => { log(`Problem reported by the page on ${who(req)} (${describeBrowser(req.headers['user-agent'])}):\n    ${body.slice(0, 1500)}`); res.writeHead(204); res.end(); });
    return;
  }
  // Piano samples (Salamander Grand Piano, installed by npm): /samples/piano/v8/C4v8.mp3
  const sm = urlPath.match(/^\/samples\/piano\/v(\d+)\/([A-G]#?\d+v\d+\.mp3)$/);
  if (sm) {
    let dir;
    try { dir = path.join(path.dirname(require.resolve(`@audio-samples/piano-mp3-velocity${sm[1]}/package.json`)), 'audio'); }
    catch { res.writeHead(404); return res.end(); }
    return fs.readFile(path.join(dir, sm[2]), (err, data) => {
      if (err) { res.writeHead(404); return res.end(); }
      res.writeHead(200, { 'Content-Type': 'audio/mpeg', 'Cache-Control': 'public, max-age=31536000, immutable' });
      res.end(data);
    });
  }
  // AI music: list models, generate phrases.
  if (urlPath === '/api/models') {
    const list = [...ai.models.values()].map(m => m.publicInfo());
    res.writeHead(200, { 'Content-Type': 'application/json', 'Cache-Control': 'no-store' });
    return res.end(JSON.stringify({ models: list }));
  }
  if (urlPath === '/api/generate') {
    const q = new URL(req.url, 'http://x').searchParams;
    const level = Math.max(1, Math.min(11, Number(q.get('level')) || 1)); // 11 = Free level
    const count = Math.max(1, Math.min(6, Number(q.get('count')) || 1));
    const t0 = now();
    const opts = {};  // keep the session's key and time signature when the app asks for them
    if (q.get('fifths') !== null && /^-?\d+$/.test(q.get('fifths'))) opts.fifths = Number(q.get('fifths'));
    if (q.get('meter') && /^\d+\/\d+$/.test(q.get('meter'))) opts.meter = q.get('meter');
    if (ai.FEELS[q.get('feel')]) opts.feel = q.get('feel');               // focused / balanced / adventurous
    const ctx = q.get('ctx');                                              // the previous stave, to continue from
    if (ctx && /^\d+(,\d+){0,700}$/.test(ctx) && opts.fifths != null && opts.meter) opts.ctx = ctx;
    ai.generate(q.get('model') || '', level, count, opts).then(phrases => {
      res.writeHead(200, { 'Content-Type': 'application/json', 'Cache-Control': 'no-store' });
      res.end(JSON.stringify({ phrases, ms: Math.round(now() - t0) }));
    }).catch(e => {
      log(`AI generation failed: ${e.message}`);
      res.writeHead(500, { 'Content-Type': 'application/json' });
      res.end(JSON.stringify({ error: e.message }));
    });
    return;
  }
  if (urlPath === '/') { urlPath = '/index.html'; log(`Page opened from ${who(req)} (${describeBrowser(req.headers['user-agent'])})`); }
  const file = path.normalize(path.join(PUBLIC_DIR, urlPath));
  if (!file.startsWith(PUBLIC_DIR + path.sep)) { res.writeHead(403); return res.end(); }
  fs.readFile(file, (err, data) => {
    if (err) { res.writeHead(404, { 'Content-Type': 'text/plain' }); return res.end('Not found'); }
    res.writeHead(200, { 'Content-Type': MIME[path.extname(file)] || 'application/octet-stream', 'Cache-Control': 'no-cache' });
    res.end(data);
  });
}
let secure = false;
function createServer() {
  const key = path.join(CERT_DIR, 'key.pem'), cert = path.join(CERT_DIR, 'cert.pem');
  if (fs.existsSync(key) && fs.existsSync(cert)) {
    secure = true;
    return https.createServer({ key: fs.readFileSync(key), cert: fs.readFileSync(cert) }, serveStatic);
  }
  return http.createServer(serveStatic);
}

/* ---------------- WebSocket ---------------- */
let wss;
function broadcast(obj) {
  const msg = JSON.stringify(obj);
  for (const c of wss.clients) if (c.readyState === WebSocket.OPEN) c.send(msg);
}
function attachSockets(server) {
  wss = new WebSocketServer({ server, path: '/ws' });
  wss.on('connection', (ws, req) => {
    ws.isAlive = true;
    ws.on('pong', () => { ws.isAlive = true; });
    log(`Browser connected from ${who(req)} (${wss.clients.size} connected)`);
    ws.send(JSON.stringify({ type: 'devices', names: midi.names, error: midi.error }));
    ws.on('message', raw => {
      let m; try { m = JSON.parse(raw); } catch { return; }
      // Clock sync: reply immediately with our time. The browser works out the offset.
      if (m.type === 'ping') ws.send(JSON.stringify({ type: 'pong', id: m.id, c0: m.c0, s: now() }));
    });
    ws.on('close', () => log(`Browser disconnected (${wss.clients.size} connected)`));
  });
  // Drop dead connections (iPad asleep, Wi-Fi gone) so they don't pile up.
  setInterval(() => {
    for (const ws of wss.clients) {
      if (!ws.isAlive) { ws.terminate(); continue; }
      ws.isAlive = false; ws.ping();
    }
  }, 10000).unref();
}

/* ---------------- MIDI ---------------- */
const midi = { lib: null, inputs: new Map(), names: [], error: null };
try { midi.lib = require('@julusian/midi'); }
catch (e) { midi.error = 'MIDI library failed to load: ' + e.message; }

// Called for every raw MIDI message. Exported so it can be tested without hardware.
function handleMidiMessage(bytes, t = now(), device = '') {
  const [status, d1, d2] = bytes, cmd = status & 0xf0;
  if (cmd === 0x90 && d2 > 0) broadcast({ type: 'note', on: true, note: d1, vel: d2, t, device });
  else if (cmd === 0x80 || (cmd === 0x90 && d2 === 0)) broadcast({ type: 'note', on: false, note: d1, t, device });
  else if (cmd === 0xb0 && d1 === 64) broadcast({ type: 'pedal', down: d2 >= 64, t });
}

function listPorts() {
  const probe = new midi.lib.Input(), out = [];
  const n = probe.getPortCount();
  for (let i = 0; i < n; i++) out.push(probe.getPortName(i));
  probe.closePort?.();
  return out;
}
const skipPort = name => /midi through|through port/i.test(name) || (MIDI_FILTER && !name.toLowerCase().includes(MIDI_FILTER));

// RtMidi has no hot-plug event, so check the port list every couple of seconds.
let skipPolls = 0, failures = 0;
function syncPorts() {
  if (!midi.lib) return;
  if (skipPolls > 0) { skipPolls--; return; }
  let ports;
  try { ports = listPorts(); failures = 0; }
  catch (e) {
    const msg = 'Could not read MIDI devices: ' + e.message;
    if (midi.error !== msg) { log(msg); midi.error = msg; if (wss) broadcast({ type: 'devices', names: [], error: msg }); }
    failures++; skipPolls = Math.min(15, failures * 2); // back off: retry less often while it keeps failing
    return;
  }
  const wanted = new Set(ports.filter(p => !skipPort(p)));
  let changed = false;
  for (const [name, input] of midi.inputs) {
    if (!wanted.has(name)) { try { input.closePort(); } catch {} midi.inputs.delete(name); log(`MIDI disconnected: ${name}`); changed = true; }
  }
  for (const name of wanted) {
    if (midi.inputs.has(name)) continue;
    const idx = ports.indexOf(name), input = new midi.lib.Input();
    input.on('message', (_delta, bytes) => handleMidiMessage(bytes, now(), name));
    input.ignoreTypes(true, true, true); // ignore sysex, timing clock, active sensing
    try { input.openPort(idx); midi.inputs.set(name, input); log(`MIDI connected: ${name}`); changed = true; }
    catch (e) { log(`Could not open MIDI input "${name}" (is another app using it?): ${e.message}`); }
  }
  midi.names = [...midi.inputs.keys()];
  midi.error = null;
  if (changed && wss) broadcast({ type: 'devices', names: midi.names, error: null });
}

/* ---------------- start ---------------- */
function lanAddresses() {
  const out = [];
  for (const list of Object.values(os.networkInterfaces())) for (const a of list || [])
    if (a.family === 'IPv4' && !a.internal) out.push(a.address);
  return out;
}
function start() {
  const server = createServer();
  // Log live-connection attempts before the WebSocket library handles them, so failed handshakes are visible too.
  server.on('upgrade', req => log(`Live connection requested by ${who(req)} for ${req.url}`));
  attachSockets(server);
  server.on('error', e => {
    if (e.code === 'EADDRINUSE') console.error(`Port ${PORT} is already in use. Close the other program or run with PORT=3001.`);
    else console.error(e);
    process.exit(1);
  });
  server.listen(PORT, HOST, () => {
    const proto = secure ? 'https' : 'http';
    console.log('\n  Flowstave is running.\n');
    console.log(`  On this computer:  ${proto}://localhost:${PORT}`);
    for (const ip of lanAddresses()) console.log(`  On your iPad:      ${proto}://${ip}:${PORT}`);
    console.log(`  Host name:         ${proto}://${os.hostname()}.local:${PORT}  (works if your network supports .local names)\n`);
    if (midi.error) console.log('  ' + midi.error + '\n');
    const loaded = ai.loadModels(MODELS_DIR);
    if (loaded.error) console.log('  AI music: ' + loaded.error + '\n');
    else if (loaded.names.length) console.log(`  AI music models: ${loaded.names.join(', ')}\n`);
    if (loaded.skipped && loaded.skipped.length) console.log(`  Skipped models: ${loaded.skipped.join('; ')}\n`);
    else console.log('  AI music: no models installed yet (see ai/README.md)\n');
    syncPorts();
    if (!midi.names.length && !midi.error) log('No MIDI keyboard found yet. Plug one in; it will be picked up automatically.');
    setInterval(syncPorts, POLL_MS).unref();
  });
  const shutdown = () => { for (const i of midi.inputs.values()) try { i.closePort(); } catch {} process.exit(0); };
  process.on('SIGINT', shutdown); process.on('SIGTERM', shutdown);
  return server;
}

if (require.main === module) start();
module.exports = { start, handleMidiMessage };
