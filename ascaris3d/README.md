# Ascaris lumbricoides – versione 3D

Ciclo vitale su anatomia reale. La pagina pronta è in `dist/`: va aperta tramite un server locale,
perché il browser non carica il modello da `file://`. Per esempio:

    cd dist && python3 -m http.server 8765   # poi http://localhost:8765

- `src/main.js`, `src/index.html`: sorgenti (three.js). `src/route3d.json`: percorso 3D.
- `build.mjs`: crea `dist/` (`npm install`, poi `node build.mjs`).
- `tools/`: estrazione del modello (`prune.mjs`, dal pacchetto npm `open-anatomy-3d@0.1.0`)
  e calcolo del percorso dalle mesh degli organi (`dump.mjs`, `route.py`, `route2.py`).
- `dist/index.html?capture`: modalità per l'esportazione video, con `window.renderAt(secondi)`.

Anatomia: BodyParts3D © The Database Center for Life Science (CC BY 4.0), via open-anatomy-3d
(CC BY 4.0), modificata (selezione delle strutture). Vedi `models/ATTRIBUTION.md`.
