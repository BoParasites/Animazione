import { NodeIO } from '@gltf-transform/core';
import { ALL_EXTENSIONS } from '@gltf-transform/extensions';
import { MeshoptDecoder } from 'meshoptimizer';
import fs from 'fs';
await MeshoptDecoder.ready;
const io = new NodeIO().registerExtensions(ALL_EXTENSIONS).registerDependencies({ 'meshopt.decoder': MeshoptDecoder });
const doc = await io.read('models/anatomy-male.glb');
const want = process.argv.slice(2);
fs.mkdirSync('/tmp/claude-0/-home-user-Animazione/eccf1370-8ebb-52f3-8aa1-59b063e81af7/scratchpad/mesh', { recursive: true });
for (const n of doc.getRoot().listNodes()) {
  const m = n.getMesh(); if (!m || (want.length && !want.includes(n.getName()))) continue;
  const wm = n.getWorldMatrix(); const V = [], F = []; let off = 0;
  for (const p of m.listPrimitives()) {
    const a = p.getAttribute('POSITION'), v = [0,0,0];
    for (let i = 0; i < a.getCount(); i++) { a.getElement(i, v);
      V.push(wm[0]*v[0]+wm[4]*v[1]+wm[8]*v[2]+wm[12], wm[1]*v[0]+wm[5]*v[1]+wm[9]*v[2]+wm[13], wm[2]*v[0]+wm[6]*v[1]+wm[10]*v[2]+wm[14]); }
    const idx = p.getIndices(); if (idx) for (let i = 0; i < idx.getCount(); i++) F.push(idx.getScalar(i) + off);
    off += a.getCount();
  }
  fs.writeFileSync(`/tmp/claude-0/-home-user-Animazione/eccf1370-8ebb-52f3-8aa1-59b063e81af7/scratchpad/mesh/${n.getName()}.json`, JSON.stringify({ V, F }));
}
