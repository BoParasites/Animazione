const { chromium } = require('playwright');
const [,, from, to, step] = process.argv.map(Number);
const FPS = 30;
(async () => {
  const b = await chromium.launch({ args: ['--enable-unsafe-swiftshader', '--use-angle=swiftshader', '--ignore-gpu-blocklist'] });
  const p = await b.newPage({ viewport: { width: 1920, height: 1080 } });
  p.on('pageerror', e => console.log('PE', e.message));
  await p.goto('http://localhost:8765/index.html?capture');
  await p.waitForFunction(() => window.__ready === true, null, { timeout: 180000 });
  await p.evaluate(() => document.fonts.ready);
  const T0 = Date.now(); let n = 0;
  for (let i = from; i < to; i += step) {
    const out = `frames/f_${String(i).padStart(5, '0')}.png`;
    if (require('fs').existsSync(out)) continue;
    await p.evaluate(s => window.renderAt(s), i / FPS);
    await p.screenshot({ path: out });
    if (++n % 50 === 0) console.log(`worker ${from}: ${n} frames, ${((Date.now() - T0) / n).toFixed(0)} ms/frame`);
  }
  await b.close(); console.log('done', from);
})();
