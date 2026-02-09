# SemesterProject – Adversarial Patch Attacks auf Verkehrsschilderkennung

Dieses Projekt untersucht **adversariale Angriffe auf Verkehrsschilderkennungssysteme** auf Basis des GTSRB-Datensatzes (German Traffic Sign Recognition Benchmark). Der Fokus liegt auf der Erzeugung adversarialer Patches im digitalen Raum und deren Übertragbarkeit in die physische Welt durch Ausdruck und erneutes Abfotografieren.

## Projektstruktur

```
SemesterProject/
├── code/                         # Gesamter Projektcode
│   ├── adversarial_patch_attack.py
│   ├── adversarial_patch_attack_split.py
│   ├── GRAPHITE_attack.py
│   ├── segment_digital_images.py
│   ├── crop_images.py
│   ├── show_adversarial_patch.ipynb
│   ├── segment.ipynb
│   ├── classify_physical_images.ipynb
│   ├── GTSRB/                    # Datensatz-Loader und OOD-Detektor
│   │   ├── gtsrb.py
│   │   ├── detectors.py
│   │   ├── labels.txt
│   │   ├── Meta.csv, Train.csv, Test.csv
│   │   └── test/                 # Testbilder
│   ├── helper_files/             # Hilfsskripte
│   │   └── resize.py
│   ├── 64x64/                    # Vortrainierte Modelle (64x64 Eingabe)
│   ├── digital_imgs/             # Originalbilder (16 Verkehrsschilder)
│   ├── digital_imgs_1200/        # Hochskalierte Bilder (1200x1200)
│   ├── sam_masks/                # SAM-Segmentierungsmasken
│   ├── phys_imgs/                # Fotos ausgedruckter Schilder
│   ├── output/                   # Ergebnisse (Bounding Boxes, Crops)
│   └── requirements.txt
├── GroundingDINO/                # Externe Bibliothek zur Objekterkennung
├── ablauf.txt                    # Projektablauf und Meeting-Notizen
└── README.md
```

## Voraussetzungen

```bash
pip install -r code/requirements.txt
```

Wichtige Abhängigkeiten: `torch`, `torchvision`, `adversarial-robustness-toolbox`, `pytorch-ood`, `transformers`, `kornia`, `foolbox`, `pillow-heif`, `pandas`, `matplotlib`

Zusätzlich wird [GroundingDINO](https://github.com/IDEA-Research/GroundingDINO) für die Objekterkennung benötigt (im Ordner `GroundingDINO/` enthalten).

## Vortrainierte Modelle

Im Ordner `code/64x64/` liegen vortrainierte Modelle für verschiedene Klassifikationsaufgaben:

| Modell                        | Aufgabe                           |
|-------------------------------|-----------------------------------|
| `label-net-resnet18-64.pt`    | Schildklasse (ResNet18)           |
| `label-net-wrn40-64.pt`       | Schildklasse (WideResNet40)       |
| `color-net-resnet18-64.pt`    | Farbklassifikation (ResNet18)     |
| `color-net-wrn40-64.pt`       | Farbklassifikation (WideResNet40) |
| `shape-net-resnet18-64.pt`    | Formklassifikation (ResNet18)     |
| `shape-net-wrn40-64.pt`       | Formklassifikation (WideResNet40) |

---

## Python-Skripte (`code/`)

### `adversarial_patch_attack.py`

Erzeugt einen **universellen adversarialen Patch** auf 16 digitalen Verkehrsschildbildern mithilfe der ART-Bibliothek (Adversarial Robustness Toolbox).

- **Angriff:** `AdversarialPatchPyTorch` (untargeted)
- **Modell:** ResNet18 (64x64)
- **Parameter:** Patchgröße 3x16x16, 800 Iterationen, LR 0.1, Skalierung 0.2–0.6, Rotation bis 45°
- **Eingabe:** 16 Bilder aus `digital_imgs/` mit SAM-Masken aus `sam_masks/`
- **Ausgabe:** `adversarial_patch_adv.pt` (adversariale Bilder, Originale, Labels)

### `adversarial_patch_attack_split.py`

Erweiterte Version des Patch-Angriffs auf einem **deutlich größeren Datensatz** (8000 Trainings- / 200 Testbilder aus GTSRB).

- **Angriff:** `AdversarialPatchPyTorch` (targeted, Zielklasse 14 – Tempo 80)
- **Modell:** ResNet18 (64x64)
- **Parameter:** Patchgröße 3x16x16, 800 Iterationen, LR 0.1, Batch-Größe 16
- **Hilfsfunktion:** `collect_samples()` – Lädt Datensatz-Samples effizient
- **Ausgabe:** `adversarial_patch_adv_split.pt` (Patch, Patch-Maske, adversariale Beispiele)

### `GRAPHITE_attack.py`

Implementiert den **GRAPHITE-Angriff** (Geometric Perturbations and Texture Injection) – eine alternative Angriffsmethode, die geometrische Transformationen mit Texturmanipulationen kombiniert.

- **Angriff:** `GRAPHITEWhiteboxPyTorch`
- **Modell:** ResNet18 (64x64)
- **Besonderheit:** Bildweise Erzeugung adversarialer Beispiele
- **Ausgabe:** `graphite_adv.pt`

### `segment_digital_images.py`

Erkennt und schneidet Verkehrsschilder aus **physischen Fotos** mithilfe von GroundingDINO automatisch zu.

- **Modell:** GroundingDINO (SwinT_OGC)
- **Prompt:** `"only the Paper with the foto on it"`
- **Schwellwerte:** Box-Threshold 0.4, Text-Threshold 0.25
- **Eingabe:** Fotos in `phys_imgs/png/`
- **Ausgabe:** Zugeschnittene Bilder in `output/crops/`, Bounding-Box-Visualisierungen in `output/bb/`

### `crop_images.py`

Einfaches Hilfsskript zum **Hochskalieren** digitaler Bilder auf 1200x1200 Pixel (bikubische Interpolation) für den Ausdruck.

- **Eingabe:** Bilder aus `digital_imgs/`
- **Ausgabe:** Skalierte Bilder in `digital_imgs_1200/`

### `helper_files/resize.py`

Funktional identisch mit `crop_images.py` – skaliert Bilder aus `digital_imgs/` auf 1200x1200 und speichert sie in `digital_imgs_1200/`.

### `GTSRB/gtsrb.py`

PyTorch-`Dataset`-Klasse zum Laden des GTSRB-Datensatzes.

- **Klasse:** `GTSRB(Dataset)` – lädt Bilder anhand von CSV-Metadaten
- **Labels:** Gibt Multi-Label-Tupel zurück: `[Klassen-ID, Farb-ID, Form-ID]`
- **Mappings:** Klassen werden automatisch Farben (`red`, `blue`, `yellow`, `white`) und Formen (`triangle`, `circle`, `square`, `octagon`, `inv-triangle`) zugeordnet

### `GTSRB/detectors.py`

Implementiert einen **logikbasierten Out-of-Distribution-Detektor** (`LogicOOD`) für Verkehrsschilder.

- **Klasse:** `LogicOOD(Detector)` – erbt von `pytorch_ood.api.Detector`
- **Prinzip:** Nutzt mehrere Attribut-Klassifikatoren (Label, Form, Farbe) und prüft deren Konsistenz anhand von Domänenwissen (z. B. bestimmte Formen haben bestimmte Farben)
- **Methoden:**
  - `get_predictions()` – Vorhersagen aller Netzwerke abrufen
  - `consistent()` – Logische Konsistenz der Vorhersagen prüfen
  - `predict()` – OOD-Score basierend auf Konsistenz und Konfidenz

---

## Jupyter Notebooks (`code/`)

### `show_adversarial_patch.ipynb`

**Visualisierung** der Ergebnisse des adversarialen Patch-Angriffs.

- Lädt `adversarial_patch_adv.pt` und zeigt Vergleich: Original vs. adversarialer Input
- Plottet Genauigkeiten über 5 Durchläufe (81,25 %–93,75 % Fehlklassifikationsrate)
- Hilfsfunktion `to_img()` zur Konvertierung von Tensoren zu darstellbaren Bildern

### `segment.ipynb`

Erzeugt **Segmentierungsmasken** für Verkehrsschilder mit Metas SAM-Modell (Segment Anything).

- **Modell:** `facebook/sam-vit-base`
- Verarbeitet digitale Bilder mit Bounding-Box-Prompts
- Erzeugt und speichert Masken als PNG in `sam_masks/`
- Die Masken können genutzt werden, um adversariale Störungen auf den Schildbereich zu beschränken

### `classify_physical_images.ipynb`

**Klassifikation** der zugeschnittenen physischen Fotos mit vortrainierten Modellen.

- Lädt zugeschnittene Bilder aus `output/crops/`
- Klassifiziert mit WideResNet40 (vortrainiert auf GTSRB)
- **Ergebnis:** 100 % Genauigkeit auf 25 zugeschnittenen Bildern (5 Bilder × 5 Schildklassen)
- Vergleich der Vorhersagen mit Ground-Truth-Labels

---

## Projektablauf

1. **Digitaler Angriff:** Erzeugung adversarialer Patches mit ART (Adversarial Patch, GRAPHITE)
2. **Ausdruck:** Hochskalierung der Bilder auf 1200x1200 und Druck auf Papier
3. **Fotografieren:** Abfotografieren der ausgedruckten Schilder aus verschiedenen Entfernungen und Winkeln
4. **Erkennung & Zuschnitt:** Automatische Detektion und Zuschnitt der Schilder aus Fotos mittels GroundingDINO
5. **Segmentierung:** Erzeugung von Masken mit SAM zur Einschränkung der Perturbationen auf den Schildbereich
6. **Evaluation:** Klassifikation der physischen Bilder und Vergleich mit digitalen Ergebnissen; OOD-Detektion mittels LogicOOD

## Ergebnisse

- Adversariale Patches erreichen digital eine **Fehlklassifikationsrate von 81–94 %**
- Korrekt zugeschnittene physische Fotos werden mit **100 % Genauigkeit** klassifiziert
- Die physische Welt wirkt als natürlicher Filter gegen digitale Perturbationen
