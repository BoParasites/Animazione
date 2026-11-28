#!/usr/bin/env node
// Export dell'animazione schistosoma_coppia.html: Chromium headless rende ogni frame
// a tempo fisso (t = n / fps) e i PNG vengono passati a ffmpeg.
//
// Richiede three@0.160.0 in locale (le richieste al CDN vengono servite da lì):
//   npm i three@0.160.0
//   node render_coppia.mjs --out schistosoma_coppia.mp4 --seconds 20 --fps 30
//   node render_coppia.mjs --stills 0,5,10 --outdir anteprime
import { chromium } from 'playwright';
import { spawn } from 'node:child_process';
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';

const here = path.dirname(fileURLToPath(import.meta.url));
const args = Object.fromEntries(process.argv.slice(2).reduce((a, v, i, all) => {
  if (v.startsWith('--')) a.push([v.slice(2), all[i + 1] && !all[i + 1].startsWith('--') ? all[i + 1] : true]);
  return a;
}, []));
const W = +(args.w || 1920), H = +(args.h || 1080), FPS = +(args.fps || 30), SECONDS = +(args.seconds || 20);
const SS = +(args.ss || 1);                      // supersampling (rende a SS× e riduce)
const threeDir = args.three || process.env.THREE_DIR || path.join(process.cwd(), 'node_modules', 'three');
if (!fs.existsSync(path.join(threeDir, 'build', 'three.module.js'))) {
  console.error('three.js non trovato in', threeDir, '(usa --three <dir> o npm i three@0.160.0)'); process.exit(1);
}

const browser = await chromium.launch({ args: ['--use-angle=swiftshader', '--enable-unsafe-swiftshader', '--ignore-gpu-blocklist'] });
const page = await browser.newPage({ viewport: { width: W * SS, height: H * SS }, deviceScaleFactor: 1 });
page.on('console', m => { if (m.type() === 'error') console.error('[page]', m.text()); });
page.on('pageerror', e => console.error('[page]', e.message));
await page.route('https://cdn.jsdelivr.net/npm/three@0.160.0/**', async route => {
  const rel = new URL(route.request().url()).pathname.replace('/npm/three@0.160.0/', '');
  await route.fulfill({ path: path.join(threeDir, rel), contentType: 'application/javascript' });
});
const extra = args.query ? '&' + args.query : '';
await page.goto(pathToFileURL(path.join(here, 'schistosoma_coppia.html')).href + `?capture&w=${W * SS}&h=${H * SS}${extra}`);
await page.waitForFunction(() => window.__ready === true, null, { timeout: 120000 });

async function grab(t) {
  const b64 = await page.evaluate((t) => { window.__renderAt(t); return document.querySelector('canvas').toDataURL('image/png').split(',')[1]; }, t);
  return Buffer.from(b64, 'base64');
}

if (args.stills) {
  const outdir = args.outdir || 'anteprime'; fs.mkdirSync(outdir, { recursive: true });
  for (const s of String(args.stills).split(',').map(Number)) {
    const t0 = Date.now(); const png = await grab(s);
    const f = path.join(outdir, `coppia_t${String(s).padStart(5, '0')}.png`); fs.writeFileSync(f, png);
    console.log(f, `${Date.now() - t0} ms`);
  }
} else {
  const out = args.out || 'schistosoma_coppia.mp4';
  const vf = SS > 1 ? ['-vf', `scale=${W}:${H}:flags=lanczos`] : [];
  const ff = spawn('ffmpeg', ['-y', '-loglevel', 'error', '-f', 'image2pipe', '-framerate', String(FPS), '-i', '-',
    ...vf, '-c:v', 'libx264', '-preset', 'slow', '-crf', '16', '-pix_fmt', 'yuv420p', '-movflags', '+faststart', out],
    { stdio: ['pipe', 'inherit', 'inherit'] });
  const n = Math.round(SECONDS * FPS), t0 = Date.now();
  for (let i = 0; i < n; i++) {
    const png = await grab(i / FPS);
    if (!ff.stdin.write(png)) await new Promise(r => ff.stdin.once('drain', r));
    if (i % 30 === 0) console.log(`frame ${i}/${n}  ${((Date.now() - t0) / (i + 1) / 1000).toFixed(2)} s/frame`);
  }
  ff.stdin.end(); await new Promise(r => ff.on('close', r));
  console.log('scritto', out);
}
await browser.close();
