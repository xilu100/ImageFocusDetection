# ImageFocusDetection

Sprache: [English](README.md) | [中文](README_CN.md) | [Deutsch](README_DE.md)

> Hinweis: Diese Datei wurde per **KI-Übersetzung** aus `README.md` erstellt.

Ein patch-basiertes Projekt zur Erkennung von Fokusbereichen und zur Schärfeklassifikation in Bildern.  
Das Projekt zerlegt Gesamtbilder in Patches fester Größe, labelt jeden Patch automatisch, trainiert mehrere Modelle (Decision Tree / Random Forest / SVM / CNN) und projiziert die Vorhersagen zur Visualisierung als Heatmap auf das Originalbild zurück.

## 1. Projektfunktionen

- Standardisierte Rohbild-Vorverarbeitung:
  - Liest Trainings-/Validierungs-Rohbilder automatisch ein und konvertiert sie in Graustufen.
  - Richtet Bildgrößen automatisch so aus, dass sie durch `patch_size` teilbar sind, um **stabile Gittersegmentierung** sicherzustellen.
  - Protokolliert die Zuordnung zwischen Original- und normalisierten Bildern (`samples_info.csv` / `valid_samples_info.csv`).
- Patch-Segmentierung und Gitterdarstellung:
  - Zerlegt jedes Bild in nicht überlappende Gitter mit fester Patch-Größe.
  - Patch-Dateinamen enthalten explizit `row/col`-Koordinaten für die Rückprojektion auf das Originalbild.
- Automatische Pseudo-Label-Erzeugung (ohne manuelle Patch-Annotation):
  - Fusioniert Laplacian-, Sobel- und FFT-Schärfesignale zu `total_score`.
  - Unterstützt Binär- und Drei-Klassen-Labelmodus; texturarme Patches werden als `-1` markiert und beim Training gefiltert.
- Multi-Modell-Training und einheitliche Experimentkonfiguration:
  - Klassisches ML: Decision Tree, Random Forest, SVM (mit Nystroem-Kernapproximation).
  - Deep Learning: leichtes CNN (unterstützt **Klassenungleichgewichts-Gewichtung**, adaptive Gerätewahl, **LMDB-Cache**).
  - Einheitliche Konfiguration von Trainingsparametern, Modellschaltern und **Sweep (Ein-Parameter-Scan)** über `Main.py`.
- Auswertung und interpretierbare Visualisierung:
  - Gibt **Accuracy**, **Classification Report** und **Confusion Matrix** auf dem Validierungsset aus.
  - Erzeugt modellweise Prediction-Overlays (pred overlay) zur direkten Sicht auf Fokus-Lokalisierung.
  - Erzeugt in der Vorverarbeitung zusätzlich Label-Overlays, Score-Heatmaps und PCA-2D/3D-Verteilungsplots pro Sample.
- Log-Parsing, Ergebnis-Paketierung und automatische Diagrammerzeugung:
  - Sammelt strukturierte Logs und vollständige Logs automatisch in ein Run-Verzeichnis.
  - Parsed Logs zu modellweisen CSVs für vergleichbare Parameter-Metrik-Tabellen.
  - Erzeugt automatisch Time/Evaluate/Loss-Diagramme für Quervergleiche.

## 2. Projektstruktur

```text
ImageFocusDetection/
├─ data/
│  ├─ raw/
│  │  ├─ train_img/                 # Trainings-Rohbilder
│  │  └─ valid_img/                 # Validierungs-Rohbilder
│  ├─ normalized/                   # Normalisierte Trainingsbilder + samples_info.csv
│  ├─ valid_normalized/             # Normalisierte Validierungsbilder + valid_samples_info.csv
│  ├─ samples/                      # Trainings-Patches
│  ├─ valid_samples/                # Validierungs-Patches
│  ├─ samples_labels/               # Trainings-Patch-Label-CSV
│  └─ valid_samples_labels/         # Validierungs-Patch-Label-CSV
├─ logs/                            # Laufzeit-Logs und paketierte Ausgaben
├─ src/
│  ├─ Main.py                       # Einheitlicher Einstieg (empfohlen)
│  ├─ preprocessing/                # Normalisierung, Segmentierung, Auto-Labeling, Visualisierung
│  ├─ training/                     # Trainingssteuerung und Modellspeicherung
│  ├─ evaluate/                     # Validierungs-Evaluierung und Vorhersage-Visualisierung
│  ├─ models/
│  │  ├─ ml/                        # Decision Tree / Random Forest / SVM
│  │  └─ dl/                        # CNN
│  └─ tools/                        # Log-Parsing, Plotting, PCA, Hilfsfunktionen
├─ requirements.txt
└─ README.md
```

## 3. Umgebungsanforderungen

- Python: empfohlen `3.10+`
- System: Windows / Linux / macOS (Pfadbeispiele sind primär Windows-basiert)
- Hardware (CNN-Training empfohlen):
  - Bevorzugt **NVIDIA GPU** mit **CUDA**-Beschleunigung.
  - Bei **Apple Silicon** (M-Serie) kann **Metal (PyTorch MPS)** für CNN-Training aktiviert werden.
  - Ohne GPU wird automatisch auf CPU zurückgefallen (deutlich langsamer).

Abhängigkeiten installieren:

```bash
pip install -r requirements.txt
```

## 4. Datenvorbereitung

Lege Trainings- und Validierungs-Rohbilder in folgende Verzeichnisse (möglichst eindeutige Dateinamen, unterstützt `jpg/jpeg/png`):

```text
data/raw/train_img
data/raw/valid_img
```

Wichtiger Hinweis: `data/raw`, `data/raw/train_img` und `data/raw/valid_img` müssen vorab manuell erstellt werden.  
Fehlt eines dieser drei Verzeichnisse, kann die Pipeline nicht ausgeführt werden.

Quellenangabe der verwendeten Rohbilder (BibTeX):

```bibtex
@inproceedings{abuolaim2020defocus,
  title={Defocus deblurring using dual-pixel data},
  author={Abuolaim, Abdullah and Brown, Michael S},
  booktitle={European Conference on Computer Vision},
  pages={111--126},
  year={2020},
  organization={Springer}
}
```

Bilddownload und Originalprojekt-Link der Publikation:  
`https://github.com/Abdullah-Abuolaim/defocus-deblurring-dual-pixel`

Standardempfehlung zur Reproduzierbarkeit:  
Verwende nach dem Download von [All images used for training/testing](https://ln2.sync.com/dl/c45358c50/view/default/10770664840008?sync_id=0#r7kpybwk-xw8hhszh-qkj249ap-y8k2344d) die Ordner `train_l` und `val_l` unter `dd_dp_dataset_png`.  
Andere Datenquellen sind ebenfalls möglich; dies dient nur der einfachen Reproduzierbarkeit.

### 4.1 Detaillierte Beschreibung des `data/`-Verzeichnisses

```text
data/
├─ raw/
│  ├─ train_img/                           # Eingabe-Trainingsbilder (vom Nutzer bereitzustellen)
│  └─ valid_img/                           # Eingabe-Validierungsbilder (vom Nutzer bereitzustellen)
├─ normalized/
│  ├─ sample1.png ...                      # Normalisierte Trainingsbilder (Graustufen + Größenanpassung)
│  └─ samples_info.csv                     # Mapping-Informationen der Trainingsnormalisierung
├─ valid_normalized/
│  ├─ valid_sample1.png ...                # Normalisierte Validierungsbilder
│  └─ valid_samples_info.csv               # Mapping-Informationen der Validierungsnormalisierung
├─ samples/
│  ├─ sample1/
│  │  ├─ sample1_0_0.png                   # Patch-Datei mit row/col im Namen
│  │  └─ ...
│  └─ sampleN/
├─ valid_samples/
│  ├─ valid_sample1/
│  │  ├─ valid_sample1_0_0.png
│  │  └─ ...
│  └─ valid_sampleN/
├─ samples_labels/
│  ├─ sample1_labels/
│  │  ├─ sample1.csv                       # Patch-Labels und Scores
│  │  ├─ sample1_label_overlay.png         # Label-Overlay
│  │  ├─ sample1_score_overlay.png         # Score-Heatmap-Overlay
│  │  ├─ sample1_pca_2d_distribution.png
│  │  └─ sample1_pca_3d_distribution.png
│  ├─ sample2_labels/
│  └─ merged_samples_labels.csv            # Zusammengeführte Trainingstabelle (vor Training auto-generiert)
└─ valid_samples_labels/
   ├─ valid_sample1_labels/
   │  ├─ valid_sample1.csv
   │  ├─ valid_sample1_label_overlay.png
   │  ├─ valid_sample1_score_overlay.png
   │  ├─ valid_sample1_pca_2d_distribution.png
   │  └─ valid_sample1_pca_3d_distribution.png
   ├─ valid_sample2_labels/
   └─ merged_valid_samples_labels.csv      # Zusammengeführte Validierungstabelle (vor Evaluierung auto-generiert)
```

### 4.2 Wichtige CSV-Felder

`normalized/samples_info.csv` und `valid_normalized/valid_samples_info.csv`:

- `filename`: normalisierter Dateiname (z. B. `sample1.png`)
- `original_size`: Originalbildgröße (`WxH`)
- `current_size`: normalisierte Größe (durch `patch_size` teilbar)
- `original_filename`: Originaldateiname (aus `data/raw/...`)

`samples_labels/*/*.csv` und `valid_samples_labels/*/*.csv`:

- `filename`: Patch-Dateiname (enthält Gitterkoordinaten)
- `lap_score`: Laplacian-Score
- `sobel_score`: Sobel-Score
- `fft_score`: FFT-Hochfrequenzanteil
- `total_score`: fusionierter Fokus-Score
- `label`: Label (`-1/0/1/2`, Bedeutung siehe 6.4)

### 4.3 Datenfluss (von Eingabe bis Training)

1. Rohbilder in `data/raw/*` -> `normalize_raw.py` -> `normalized/valid_normalized`
2. Normalisierte Bilder -> `segment_nor_img.py` -> `samples/valid_samples`
3. Patches -> `label_patches.py` -> `samples_labels/valid_samples_labels`
4. Label-CSV zusammenführen -> `merged_samples_labels.csv` / `merged_valid_samples_labels.csv`
5. Training liest `merged_samples_labels.csv`, Evaluierung liest `merged_valid_samples_labels.csv`

## 5. Ausführung

Im Projekt-Root ausführen:

```bash
python src/Main.py
```

`Main.py` läuft standardmäßig in folgender Reihenfolge:

1. Vorverarbeitung (Normalisierung -> Patch-Schnitt -> Auto-Labeling -> Label-Visualisierung)
2. Training (gemäß Modellschaltern)
3. Evaluierung (Metriken und Vorhersage-Visualisierung auf dem Validierungsset)
4. Paketierung (Logs zu CSV, automatische Plots, Gesamtausgabe)

## 6. Konfiguration (`src/Main.py`)

### 6.1 Experimentparameter `get_experiment_config()`

- `training.patch_size`: Patch-Kantenlänge, häufig `16/32/64/128`.
- `training.sharp_threshold` / `training.blur_threshold`: **Label-Schwellen** (0~1).
- `training.PCA_components`: PCA-Parameter, `-1` bedeutet PCA aus.
- `training.sample_percentage`: Trainings-Sampling-Anteil (0~100].
- `models.*`: Modell-Hyperparameter (DT/RF/SVM/CNN).

### 6.1.1 Detaillierte Modellparameter (`models`)

#### A. Decision Tree (`models.decision_tree`)

- `max_depth`: maximale Baumtiefe.
- `min_samples_split`: minimale Stichprobe für Knotensplit.
- `min_samples_leaf`: minimale Stichprobe im Blatt.
- `class_weight`: Klassengewichtung (z. B. `balanced`).
- `random_state`: Seed für Reproduzierbarkeit.

#### B. Random Forest (`models.random_forest`)

- `n_estimators`: Anzahl Bäume.
- `max_depth`: maximale Tiefe pro Baum.
- `random_state`: Seed.
- `class_weight`: Klassengewichtung (typisch `balanced_subsample`).
- `n_jobs`: Parallel-Threads, `-1` = möglichst alle CPU-Kerne.

#### C. SVM (`models.svm`, implementiert als `StandardScaler + Nystroem + LinearSVC`)

- `nystroem_components`: Dimension der Kernapproximation.
- `nystroem_kernel`: Kernfunktion (`rbf/cosine/poly/sigmoid`).
- `nystroem_gamma`: Kernparameter (vor allem für `rbf/poly/sigmoid`).
- `random_state`: Seed (für Nystroem-Sampling).
- `svc_c`: Strafparameter C.
- `class_weight`: Klassengewichtung.
- `max_iter`: maximale Iterationen.

#### D. CNN (`models.cnn`)

- `epochs`: Anzahl Epochen.
- `batch_base`: Basis-Batchgröße; tatsächliche Batchgröße wird geräteabhängig angepasst.
- `seed`: Zufallsseed.
- `learning_rate`: Adam-Lernrate.
- `use_weighted_sampler`: gewichtetes Sampling aktivieren (`auto/true/false`).
- `sampler_weight_power`: Exponent für Sampler-Gewichte.
- `loss_weight_power`: Exponent für Loss-Klassengewichte.
- `build_lmdb_if_missing`: **LMDB-Cache** aufbauen (aktuelle Implementierung baut neu).
- `assume_fixed_size`: Patch-Größe muss strikt `patch_size` entsprechen.
- `noise_std` (optional): Stärke des Gauß-Rauschens als Augmentation.

### 6.1.2 CNN-Trainingshinweise (für Tuning)

- Netzwerkstruktur skaliert mit `patch_size`:
  - `<=16`: 1 Conv-Block
  - `<=64`: 2 Conv-Blöcke
  - `>64`: 3 Conv-Blöcke
- Verlustfunktion: `CrossEntropyLoss` mit automatisch konstruierten Klassengewichten.
- Gerätepriorität: `CUDA > MPS > CPU`; AMP, `batch_size` und `num_workers` werden automatisch gesetzt.
- Dabei:
  - `CUDA` entspricht NVIDIA-GPU-Beschleunigung (empfohlen).
  - `MPS` entspricht Apple-Silicon-Metal-Beschleunigung.
- Bei aktiviertem LMDB liegt der Cache unter `data/samples_labels/patches_ps<patch_size>.lmdb`.

### 6.2 Ausführungsschalter `get_control_config()`

- `pipeline.preprocess`: Vorverarbeitung ein/aus.
- `pipeline.train_evaluate`: Training + Evaluierung ein/aus.
- `models.decision_tree/random_forest/svm/cnn`: Modellweise ein/aus.

### 6.3 Sweep (Parameterscan)

Wird ein Parameter als Liste gesetzt, startet **Sweep** (z. B. `patch_size: [16, 32, 64]`).  
Es ist nur **ein** Listenparameter erlaubt; mehrere führen zu Fehlern.

Wichtig: Vor jedem Training wird `src/training/model_save/` geleert und neu geschrieben.  
Daher bleiben im Sweep-Modus **nur die Modellgewichte des letzten Runs** erhalten; frühere werden überschrieben.

### 6.4 Label-Modus

- Drei-Klassen-Modus: `sharp_threshold > blur_threshold`
  - `1`: scharf
  - `2`: Zwischenbereich
  - `0`: unscharf
  - `-1`: texturarm (beim Training gefiltert)
- Binärmodus: `sharp_threshold == blur_threshold`
  - `1`: scharf
  - `0`: unscharf
  - `-1`: texturarm (gefiltert)

## 7. Ausgabebeschreibung

### 7.1 Modellgewichte

Werden nach dem Training gespeichert unter:

```text
src/training/model_save/
```

Typische Dateien:

- `decision_tree_model.joblib` / `decision_tree_pca.joblib`
- `random_forest_model.joblib` / `random_forest_pca.joblib`
- `svm_model.joblib` / `svm_pca.joblib`
- `cnn_model.pth`

Hinweis: Da der Trainingseinstieg `model_save` jedes Mal neu erstellt, entsprechen diese Dateien im Sweep-Modus dem **letzten Sweep-Wert**.

### 7.2 Logs und Paketierungsverzeichnis

Jeder Lauf erzeugt ein Paketverzeichnis unter `logs/`, typische Namen:

- `NM_<timestamp>`: normaler Einzellauf
- `TR_<param>_<timestamp>` / `DT_...` / `RF_...` / `SVM_...` / `CNN_...`: Sweep-Lauf

Typische Struktur:

```text
logs/
└─ NM_2026_0423_222346/                      # paketierte Gesamtausgabe eines Laufs (Beispiel)
   ├─ 20260423_2222.log                      # strukturiertes Log (print_and_save / save)
   ├─ 20260423_2222_complete.log             # vollständiges stdout/stderr-Log
   ├─ Decision_Tree_<control>.csv            # Decision-Tree-Metrikübersicht
   ├─ Random_Forest_<control>.csv            # Random-Forest-Metrikübersicht
   ├─ SVM_<control>.csv                      # SVM-Metrikübersicht
   ├─ CNN_<control>.csv                      # CNN-Metrikübersicht
   ├─ plots/                                 # automatisch aus CSV erzeugte Diagramme
   │  ├─ <Model>_<control>_Time.png
   │  ├─ <Model>_<control>_Evaluate.png
   │  └─ CNN_<control>_Loss_*.png
   └─ predict_images/                        # Vorhersage-Overlays des Validierungssets
      └─ valid_sample*_predict_images[_tag]/
         └─ valid_sample*_pred_overlay_<DT|RF|SVM|CNN>.png
```

Dabei gilt:

- `<control>` ist der **Sweep-Steuerparameter** (ohne Sweep meist `normal`).
- `[_tag]` erscheint nur im **Sweep**-Modus zur Unterscheidung verschiedener Werte.
- `logs/_evaluate_cache` ist ein temporäres Evaluierungsverzeichnis und wird nach der Paketierung bereinigt.

Das Verzeichnis enthält typischerweise:

- modellweise Zusammenfassungs-CSVs (aus Logs geparsed)
- `plots/`-Diagramme (Zeitmetriken, Evaluierungsmetriken, CNN-Loss-Kurven)
- `predict_images/`-Visualisierungen (nach Samples gruppiert)
- Lauf-Log und Complete-Log

### 7.3 Visualisierungsergebnisse

In der Vorverarbeitung erzeugt:

- Label-Overlay (`*_label_overlay.png`)
- Score-Heatmap-Overlay (`*_score_overlay.png`)
- PCA-2D/3D-Verteilungsdiagramme (pro Sample)

In der Evaluierung erzeugt:

- Vorhersage-Overlays des Validierungssets (pro Modell, z. B. `*_pred_overlay_DT.png`)

## 8. Häufige Fragen

### 8.1 Warum werden bei jeder Vorverarbeitung einige Verzeichnisse geleert?

`Main.py` löscht und erstellt folgende Zwischenverzeichnisse neu, um saubere und reproduzierbare Läufe sicherzustellen:

- `data/samples`
- `data/samples_labels`
- `data/valid_samples`
- `data/valid_samples_labels`

### 8.2 Was tun, wenn die Ausführung zu langsam ist?

- Nur ein Modell aktivieren (über `get_control_config()`).
- `sample_percentage` reduzieren (z. B. 30~50).
- `patch_size` reduzieren oder weniger Sweep-Werte verwenden.
- CNN-`epochs` reduzieren.

### 8.3 Kann ich nur plotten oder nur Logs parsen?

Ja, relevante Skripte:

- `src/tools/log_tools.py`
- `src/tools/plot_csv.py`

### 8.4 Warum erscheint `label=-1`?

`-1` bedeutet **texturarmer** Patch (niedrige Varianz und niedriger Gradient), also „nicht zuverlässig klassifizierbar“.  
Beim Zusammenführen der Train/Eval-CSV werden `label=-1` automatisch gefiltert.

### 8.5 Wie wechsle ich zwischen Binär- und Drei-Klassenmodus?

- Binär: `sharp_threshold == blur_threshold`
- Drei-Klassen: `sharp_threshold > blur_threshold`

Wenn `blur_threshold > sharp_threshold` gesetzt ist, bricht das Programm mit Fehler ab.

### 8.6 Warum bleibt nach Sweep nur ein Modellsatz übrig?

Vor jedem Lauf wird `src/training/model_save/` geleert.  
Daher bleiben nach Sweep nur die Gewichte des letzten Runs.

### 8.7 Warum wird bei der Evaluierung ein Modell übersprungen (missing model files)?

Mögliche Ursachen:

- Modell in `get_control_config().models` deaktiviert
- letzter Trainingslauf nicht erfolgreich abgeschlossen
- Modell durch späteren Sweep-Run überschrieben

### 8.8 Warum ist `predict_images` nach Samples statt nach Modellen organisiert?

Die aktuelle Implementierung gruppiert nach Validierungs-Sample; jedes Sample-Verzeichnis enthält Overlays aller Modelle.  
So lassen sich Modellunterschiede auf demselben Bild leichter vergleichen.

### 8.9 Was ist `logs/_evaluate_cache`?

Ein temporärer Cache während der Evaluierung für Vorhersage-Overlays.  
Nach dem Packaging in `logs/<run_dir>/predict_images` wird er automatisch bereinigt.

### 8.10 Warum gibt es keinen Test-Set-Workflow?

Der Hauptworkflow ist aktuell auf `train + valid evaluate` festgelegt.  
Bei Bedarf kann analog zu `data/raw/valid_img` ein `test_img`-Pfad ergänzt werden.

### 8.11 Wie soll `PCA_components` gesetzt werden?

- `-1`: PCA deaktivieren (Original-Pixelfeatures).
- `0~1` (z. B. `0.95`): Hauptkomponenten über **kumulierte erklärte Varianz** behalten.
- Wenn der Wert größer/gleich der Originaldimension ist, wird PCA automatisch übersprungen.

### 8.12 SVM ist langsam oder konvergiert nicht. Was tun?

- `nystroem_components` reduzieren
- `svc_c` moderat verkleinern
- `max_iter` erhöhen
- oder `sample_percentage` reduzieren

### 8.13 CNN-Speicher reicht nicht aus. Was tun?

- `models.cnn.batch_base` reduzieren
- `patch_size` reduzieren
- `epochs` reduzieren
- bei Bedarf nur CNN laufen lassen

### 8.14 Warum schwanken Ergebnisse bei gleichen Parametern leicht?

Seeds sind gesetzt, aber unterschiedliche Backends/Geräte (CPU/CUDA/MPS) und parallele Low-Level-Ausführung können kleine numerische Unterschiede erzeugen.  
Für strikte Vergleichbarkeit: Gerät, Daten und Parameter fixieren.

## 9. Abhängigkeiten

Siehe `requirements.txt`:

- `numpy`
- `pandas`
- `scikit-learn`
- `joblib`
- `opencv-python`
- `torch`
- `lmdb`
- `matplotlib`

## 10. Entwicklungsvermerk

Teile dieses Projekts wurden mit CodeX-Unterstützung erstellt, einschließlich (nicht abschließend):

- großskaliges Regex-Parsing und strukturierte Log-Extraktion (z. B. `src/tools/log_tools.py`)
- vereinheitlichte Log-Sammlung und Ausgabepaketierung (z. B. `src/tools/log.py`, `src/Main.py`)
- strikte Typbindung und Strukturdefinitionen im Stil von Java / C / C++

Zusatzhinweis: Die deutsche Fassung wurde per KI übersetzt. Meine Deutschkenntnisse sind begrenzt, vielen Dank für Ihr Verständnis.

## 11. Urheberrecht und Drittanbieterhinweise (konsolidiert)

- **CodeX**: CodeX ist ein von OpenAI bereitgestelltes Tool zur Codegenerierung und Programmierassistenz.
- **Open-Source-Abhängigkeiten Dritter**: Die verwendeten Bibliotheken (`numpy`, `pandas`, `scikit-learn`, `joblib`, `opencv-python`, `torch`, `lmdb`, `matplotlib`) unterliegen den jeweiligen Lizenzen ihrer Maintainer/Communitys.
- **Datensätze und Bildmaterial**: Rohbilder unter `data/raw/*` bleiben urheberrechtlich bei den ursprünglichen Autoren/Anbietern; Nutzung/Verteilung/Publikation nur mit entsprechender Berechtigung.
- **Bildquelle dieses Projekts (BibTeX)**:
  ```bibtex
  @inproceedings{abuolaim2020defocus,
    title={Defocus deblurring using dual-pixel data},
    author={Abuolaim, Abdullah and Brown, Michael S},
    booktitle={European Conference on Computer Vision},
    pages={111--126},
    year={2020},
    organization={Springer}
  }
  ```
- **Methoden- und Begriffszitate**: In Dokumentation und Code genannte Algorithmen/Modelle (z. B. PCA, SVM, Random Forest, CNN, Nystroem) sind allgemeine Fachbegriffe; zugehörige Publikationen liegen beim jeweiligen Originalverlag bzw. den Autoren.
