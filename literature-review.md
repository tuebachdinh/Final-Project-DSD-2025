# Related Work on ML-based cfPWV Estimation from Wrist Signals

Below is a summary of key recent literature on estimating carotid-femoral pulse wave velocity (cfPWV) or vascular stiffness using peripheral (wrist/finger) signals and machine learning. Each study explores a different approach, signal type, or data source relevant to this project.

---

## 1. Garcia et al. (2022): Spectrogram + ML for cfPWV

- **Approach:** Transforms single-cycle PPG or BP signals (finger, wrist, brachial) into spectrogram images, extracts image features (energy, moments), and uses MLP/SVR regression to estimate cfPWV.
- **Similarity:** Peripheral signal + ML estimation of cfPWV; also explores noise robustness.
- **Difference:** Uses image-based feature extraction rather than time-domain analysis; only one cardiac cycle and no real-world wrist data.
- **Link:** [A learning-based image processing approach for pulse wave velocity estimation using spectrogram and machine learning](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC10020726/)

---

## 2. Li et al. (2022): XGBoost on Wrist PPG

- **Approach:** Applies XGBoost regression on morphological features from wrist PPG to estimate cfPWV.
- **Similarity:** Wrist PPG, feature-engineered ML regression to predict stiffness.
- **Difference:** Uses advanced ensemble methods; scope and dataset (real vs synthetic) not always clear.
- **Link:** [An XGBoost-based model for assessment of aortic stiffness from wrist photoplethysmogram](https://pubmed.ncbi.nlm.nih.gov/36150230/)

---

## 3. Smartwatch cfPWV Estimation (2022) – Frontiers in Cardiovascular Medicine

- **Approach:** Evaluates off-the-shelf smartwatches for cfPWV estimation in clinical settings, with high accuracy (CCC ~0.90).
- **Similarity:** Wrist wearable-based assessment aligns with wearable/real-world ambitions.
- **Difference:** Focuses on commercial devices and clinical practicality, not algorithm development.
- **Link:** [Photoplethysmography-based heart–finger pulse wave velocity assessments with consumer smartwatches](https://www.frontiersin.org/articles/10.3389/fcvm.2023.1108219/full)

---

## 4. WF-PPG Dataset (2025, in press/preprint)

- **Approach:** Dataset containing wrist and fingertip PPG signals under varying contact pressures, with ECG and BP references.
- **Similarity:** Provides real-world wrist PPG data—directly relevant to proposed wearable device integration.
- **Difference:** Focuses on signal collection & variability; does not directly predict cfPWV.
- **Link:** *No direct publication link yet—add when available.*

---

## 5. Smartphone PPG for Vascular Aging (2020)

- **Approach:** Uses ML and CNNs on smartphone PPG to estimate vascular age, a correlate of arterial stiffness.
- **Similarity:** Peripheral PPG + ML/DL for vascular health/aging.
- **Difference:** Predicts vascular age (not cfPWV directly).
- **Link:** [Photoplethysmogram based vascular aging assessment using deep convolutional neural networks](https://www.nature.com/articles/s41598-022-15240-4)

---

## 6. In-vivo Wrist PPG ML for Inter-Beat Intervals (2023)

- **Approach:** ML predicts inter-beat intervals (IBI) from wrist PPG and derivatives + demographics.
- **Similarity:** Wrist PPG analysis, time-domain feature extraction.
- **Difference:** Focuses on heart rate (IBI), not arterial stiffness.
- **Link:** *No direct publication link found—add when available.*

---

## 7. Jin et al. (2021): cfPWV from Radial Pressure (PLoS ONE)

- **Approach:** Uses Gaussian Process Regression on extracted features and RNN on full radial pressure waveforms, tested on both Twins UK and synthetic PWDB data.
- **Similarity:** Peripheral waveform + feature-based and end-to-end ML for cfPWV; synthetic and human data.
- **Difference:** Uses pressure (applanation tonometry), not free-hand wrist PPG.
- **Link:** [Non-invasive assessment of arterial stiffness using radial pressure wave and machine learning: Application in the Twins UK cohort and in-silico data](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0256551)

---

## 🧭 Comparison Table

| Study                | Signal Type        | Dataset            | ML Approach          | Device Context       | Relation to This Work                         |
|----------------------|-------------------|--------------------|----------------------|---------------------|-----------------------------------------------|
| Garcia et al.        | PPG/BP Spectrogram| Synthetic          | MLP/SVR (Image)      | Lab/Bench           | Spectrogram features, ML regression           |
| Li et al.            | Wrist PPG         | WRIST PPG          | XGBoost              | Potential wearable  | Morphological features, ensemble ML           |
| Smartwatch studies   | Wrist PPG         | Real Clinical      | Various              | Commercial wearable | Closest to real-world device                  |
| WF-PPG dataset       | Wrist/Finger PPG  | Real device        | Data resource        | Wearable            | Data platform for signal variation            |
| Smartphone PPG       | Finger PPG        | Real-world         | ML/CNN               | Smartphone          | ML/DL on peripheral PPG, vascular health      |
| IBI study            | Wrist PPG         | In-vivo            | ML                   | Wearable            | HR/IBI focus                                 |
| Jin et al.           | Radial Pressure   | Synthetic + Twins  | GPR + RNN            | Clinic/Bench        | Closest algorithmic/validation match          |

---

## 🔍 Highlights & Project Novelty

- Most prior work uses either synthetic or clinical radial pressure, or finger/wrist PPG, but rarely both synthetic **and** real wrist PPG in a unified ML pipeline.
- **This project stands out by:**
    - Combining two synthetic datasets (HaeMod + PWDB)
    - Integrating real-world wrist PPG from wearables
    - Evaluating both feature-based and end-to-end ML (possibly including spectrograms)
    - Including explicit noise/artifact modeling for wrist wearables

---

