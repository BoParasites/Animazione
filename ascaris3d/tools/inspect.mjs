import { NodeIO } from '@gltf-transform/core';
import { ALL_EXTENSIONS } from '@gltf-transform/extensions';
import { MeshoptDecoder } from 'meshoptimizer';
await MeshoptDecoder.ready;
const io = new NodeIO().registerExtensions(ALL_EXTENSIONS).registerDependencies({ 'meshopt.decoder': MeshoptDecoder });
const doc = await io.read('models/anatomy-male.glb');
const root = doc.getRoot();
for (const n of root.listNodes()) {
  const m = n.getMesh(); if (!m) { console.log('node', n.getName()); continue; }
  let mn=[1e9,1e9,1e9], mx=[-1e9,-1e9,-1e9], tris=0;
  const wm = n.getWorldMatrix();
  for (const p of m.listPrimitives()) { const a=p.getAttribute('POSITION'); tris+= (p.getIndices()?.getCount()||a.getCount())/3;
    const v=[0,0,0]; for(let i=0;i<a.getCount();i++){a.getElement(i,v);
      const x=wm[0]*v[0]+wm[4]*v[1]+wm[8]*v[2]+wm[12], y=wm[1]*v[0]+wm[5]*v[1]+wm[9]*v[2]+wm[13], z=wm[2]*v[0]+wm[6]*v[1]+wm[10]*v[2]+wm[14];
      const w=[x,y,z]; for(let k=0;k<3;k++){mn[k]=Math.min(mn[k],w[k]);mx[k]=Math.max(mx[k],w[k]);}}}
  console.log(n.getName().padEnd(22), tris|0, mn.map(v=>v.toFixed(3)).join(','), '|', mx.map(v=>v.toFixed(3)).join(','));
}
