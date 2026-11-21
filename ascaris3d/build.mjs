import { build } from 'esbuild';
import fs from 'fs';
fs.mkdirSync('dist/models', { recursive: true });
await build({ entryPoints: ['src/main.js'], bundle: true, minify: true, format: 'iife', target: 'es2020', outfile: 'dist/app.js', loader: { '.json': 'json' }, legalComments: 'eof' });
fs.copyFileSync('src/index.html', 'dist/index.html');
fs.copyFileSync('models/anatomy-ascaris.glb', 'dist/models/anatomy-ascaris.glb');
console.log('ok', fs.statSync('dist/app.js').size);
