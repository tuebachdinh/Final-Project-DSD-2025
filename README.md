# Machine Learning Analysis of Wrist Pulse Waveforms for Arterial Stiffness Estimation

## Project Goal

This project aims to **quantify and predict arterial stiffness from wrist pulse waveforms**—focusing on both simulated pressure and photoplethysmogram (PPG) signals—using classical and machine learning methods.  
Our broader objective is to understand how arterial stiffness (central pulse wave velocity, PWV) is reflected in wrist pulse morphology, and to lay the groundwork for **practical, non-invasive vascular health monitoring with consumer wearable devices**.

---

## Motivation

- **Clinical importance:** Arterial stiffness (cfPWV) is a critical marker for cardiovascular risk, but direct measurement requires clinical-grade equipment.
- **Wearable potential:** Wrist-based pulse signals (PPG/pressure) can be easily acquired by smartwatches and fitness bands.
- **Scientific challenge:** It is not fully understood how central stiffness affects peripheral waveform shape, especially in the presence of noise, real-life variation, or consumer hardware constraints.

---

## Datasets

### 1. HaeMod Virtual Subjects (Willemet et al., Am J Physiol, 2015)
- **3,325 virtual subjects** with simulated pressure, flow, and area waveforms at multiple arterial locations, including the wrist (radial artery).
- Provided simulation parameters and computed indices (cfPWV, cardiac output, geometry).
- Download and documentation: [http://haemod.uk/original](http://haemod.uk/original)
- Reference:  
  Willemet M, Chowienczyk P, Alastruey J.  
  *A database of virtual healthy subjects to assess the accuracy of foot-to-foot pulse wave velocities for estimation of aortic stiffness*.  
  Am J Physiol Heart Circ Physiol, 2015. [Link](https://doi.org/10.1152/ajpheart.00175.2015)

### 2. PWDB In Silico Database (Charlton et al., Am J Physiol, 2019)
- **4,374 virtual subjects** with simulated pressure, flow, area, and PPG waveforms at multiple sites (including wrist PPG).
- Includes demographic and hemodynamic diversity (ages 25–75).
- Download: [https://github.com/peterhcharlton/pwdb](https://github.com/peterhcharlton/pwdb)
- Reference:  
  Charlton PH, Mariscal Harana J, Vennin S, Li Y, Chowienczyk P, Alastruey J.  
  *Modeling arterial pulse waves in healthy aging: a database for in silico evaluation of hemodynamics and pulse wave indexes*.  
  Am J Physiol Heart Circ Physiol, 2019. [Link](https://doi.org/10.1152/ajpheart.00218.2019)

---

## Project Workflow

### 1. Data Exploration & Preparation
- Load simulated pulse waveforms (pressure or PPG) at the wrist for all subjects.
- Filter for physiologically plausible signals.
- Visualize waveforms across different stiffness (PWV) levels.

### 2. Feature Engineering & Classical Analysis
- Extract morphological features: pulse height, upstroke/downstroke times, slopes, area-under-curve, systolic/diastolic times, and classical PPG indices (Reflection Index, Stiffness Index, Modified Aging Index).
- Correlate features with cfPWV.
- Visualize how waveform shape encodes arterial stiffness.

### 3. Machine Learning Modeling
- **Feature-based regression:** Use extracted features as input for linear or non-linear models to predict cfPWV.
- **Waveform-driven regression:** Use the entire waveform as input for ML (e.g., regression trees, random forests, shallow neural nets).
- Benchmark ML performance vs. classical index-based approaches.

### 4. Interpretation & Insight
- Analyze model accuracy, interpretability, and key predictive features.
- Discuss practical implications for wearable/consumer devices.

### 5. **Real-World Extension (Future)**
- Integrate and test with real wrist PPG/pressure signals from actual hardware (e.g., smartwatches, custom PPG sensors).
- Assess model robustness to noise, movement artifacts, skin tone, sensor placement, and individual variation.
- Explore calibration strategies and transfer learning to bridge the gap between in silico and real-world data.

---

## Enhancement Opportunities

- **Cross-database validation:** Train on one synthetic dataset, test on the other, analyze transferability of features and models.
- **End-to-end deep learning:** Try CNNs or RNNs for direct waveform-to-stiffness mapping, especially when large-scale real data is available.
- **Domain adaptation:** Adapt in silico-trained models to real wrist data using domain adaptation or transfer learning.
- **Noise robustness:** Test models on signals with added synthetic noise and/or real-world PPG noise/artifacts.
- **Device-specific tuning:** Study and compensate for characteristics of specific wearable sensors.
- **Personalization:** Investigate individualized models or calibration routines for subject-specific arterial property estimation.

---

## How to Use

1. Download either database (see links above).
2. Use provided Matlab scripts to extract features, process waveforms, and visualize results.
3. Export features and labels (e.g., `radial_features_labels.csv`) for downstream ML in Python, MATLAB, or other frameworks.
4. Use the provided Jupyter/Python notebooks as a template for model development, evaluation, and visualization.
5. For real-world data, adapt scripts for your device and data format, and follow the same analysis pipeline.

---

## Acknowledgements

- Willemet M, Chowienczyk P, Alastruey J (2015).  
  [https://doi.org/10.1152/ajpheart.00175.2015](https://doi.org/10.1152/ajpheart.00175.2015)
- Charlton PH, Mariscal Harana J, Vennin S, Li Y, Chowienczyk P, Alastruey J (2019).  
  [https://doi.org/10.1152/ajpheart.00218.2019](https://doi.org/10.1152/ajpheart.00218.2019)

---

## Contact

Questions, suggestions, or collaboration ideas?  
Contact: Dinh Bach Tue / tue.dinh@aalto.fi

---

**Discussion and collaboration are welcome!**  
We encourage contributions, feature ideas, and new datasets to help bring in silico insights into practical, real-world cardiovascular health monitoring.
