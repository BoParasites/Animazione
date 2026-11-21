import { NodeIO } from '@gltf-transform/core';
import { ALL_EXTENSIONS, EXTMeshoptCompression } from '@gltf-transform/extensions';
import { prune } from '@gltf-transform/functions';
import { MeshoptDecoder, MeshoptEncoder } from 'meshoptimizer';
await MeshoptDecoder.ready; await MeshoptEncoder.ready;
const io = new NodeIO().registerExtensions(ALL_EXTENSIONS).registerDependencies({ 'meshopt.decoder': MeshoptDecoder, 'meshopt.encoder': MeshoptEncoder });
const doc = await io.read('models/anatomy-male.glb');
const keep = new Set(`skin skull mandible teeth spine_cervical spine_thoracic spine_lumbar ribs_l ribs_r sternum clavicle_l clavicle_r scapula_l scapula_r
humerus_l humerus_r forearm_l forearm_r hand_l hand_r pelvis femur_l femur_r patella_l patella_r lower_leg_l lower_leg_r foot_l foot_r
brain eye_l eye_r lung_l lung_r heart liver gallbladder stomach spleen pancreas small_intestine colon appendix esophagus trachea bronchi pharynx tongue
kidney_l kidney_r adrenal_l adrenal_r ureter_l ureter_r bladder aorta vena_cava_inferior vena_cava_superior veins_abdomen pulmonary_vessels diaphragm`.split(/\s+/));
for (const n of doc.getRoot().listNodes()) if (n.getMesh() && !keep.has(n.getName())) n.dispose();
await doc.transform(prune());
doc.createExtension(EXTMeshoptCompression).setRequired(true).setEncoderOptions({ method: EXTMeshoptCompression.EncoderMethod.QUANTIZE });
await io.write('models/anatomy-ascaris.glb', doc);
console.log('nodes', doc.getRoot().listNodes().filter(n => n.getMesh()).length);
