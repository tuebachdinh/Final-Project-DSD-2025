# Comprehensive Analysis Pipeline for Arterial Stiffness Estimation from Wrist Pulse Signals

## Overview

A comprehensive pipeline for predicting arterial stiffness from wrist pulse waveforms using data visualization, signal processing, classical machine learning, advanced deep learning pipeline, and model-interpretability techniques.

**Research Team:** 
- Dinh Bach Tue (Student) - Aalto University, Finland
- PhD Researcher Bahmani Nima (Advisor)
- Professor Stephan Sigg (Supervisor)

**Objective:** Develop robust foundation for transferring knowledge from synthetic waveform data to real wearable device data for non-invasive cardiovascular health monitoring.

## Datasets

### 1. PWDB In Silico Database (Charlton et al., 2019) - **Main Dataset**
- **4,374 virtual subjects** with simulated **P (pressure), U (flow), A (area), and PPG waveforms**
- **Multi-site measurements** including radial artery (wrist focus)
- **Demographic diversity**: Ages 25–75 years
- **Target variable**: Carotid-femoral PWV (cfPWV) for arterial stiffness
- Download: [https://zenodo.org/records/3275625](https://zenodo.org/records/3275625)
- Filename : 'pwdb_data.mat' - 701.7 MB

### 2. HaeMod Virtual Subjects (Willemet et al., 2015) - **Archived**
- **3,325 virtual subjects** with P, U, A waveforms (**no PPG**)
- Initial exploration dataset, results in `archieved/` folder
- Download: [http://haemod.uk/original](http://haemod.uk/original)
---

## Repository Structure

```
Final-Project-DSD-2025/
├── src/                     # 12-part analysis pipeline
├── utils/                   # Signal processing & ML utilities
├── data/                    # Dataset storage
├── models/                  # Trained networks & results
├── images/                  # Generated visualizations
├── tables/                  # Performance metrics
├── docs/                    # Detailed research reports
└── archieved/               # Initial work (2015 dataset)
```

## Workflow

**12-Part Analysis Pipeline:**
- **Parts 1-3**: Data preparation, visualization, PTT analysis
- **Parts 4-6**: Classical ML, signal processing, feature extraction
- **Parts 7-9**: Data augmentation, deep learning (CNN/GRU/TCN)
- **Parts 10-12**: Model interpretability, cross-validation, evaluation

---

## How to Use

### Prerequisites
- MATLAB R2025a with Toolbox
   + Signal Processing Toolbox
   + Deep Learning Toolbox
   + Curve Fitting Toolbox
   + Mapping Toolbox
   + Statistics and Machine Learning Toolbox
- PWDB dataset (download `pwdb_data.mat` from [Zenodo](https://zenodo.org/records/3275625))

### Quick Start
1. **Setup Data**: Place `pwdb_data.mat` in `data/exported_data/`
2. **Run Analysis**: Execute `src/main.m` or individual parts
3. **View Results**: Check `images/` for generated plots and `models/` for trained networks

Note: If the current plots or models are not removed, running the script will override the images(.png) and the models(.mat)


### Output Structure
- **Figures**: `images/part[N]_[description].png`
- **Models**: `models/part9_cnn_gru_models.mat` - example
- **Analysis**: Interpretability results in `.mat` files

---


## Future Directions
This project serves as a baseline for transferring models from synthetic to real wearable device data. Several avenues for future work include:
- **Multi-sensor integration**: Extend models to fuse PPG, mmWave, and other wrist-based signals for richer arterial stiffness estimation.
- **Transfer learning & domain adaptation**: Apply fine-tuning and adaptation strategies to bridge the gap between synthetic in silico data and real-world wearable measurements.
- **Data augmentation studies**: Investigate how different augmentation strategies affect model robustness, particularly in noisy or low-quality signals.
- **Model interpretability**: Explore the effect of window resolution in occlusion and perturbation analyses to better understand which waveform segments drive predictions.
- **Architectural exploration**: Benchmark additional architectures beyond CNN, GRU, and TCN, including Transformers and hybrid models, to evaluate their suitability for pulse waveform analysis.

Ultimately, the goal is to create a pipeline that is not only accurate on synthetic datasets but also generalizes effectively to wearable-collected signals for practical, non-invasive cardiovascular health monitoring.

## References

- Charlton PH, Mariscal Harana J, Vennin S, Li Y, Chowienczyk P, Alastruey J. *Modeling arterial pulse waves in healthy aging: a database for in silico evaluation of hemodynamics and pulse wave indexes*. Am J Physiol Heart Circ Physiol, 2019. [DOI](https://doi.org/10.1152/ajpheart.00218.2019)
- Willemet M, Chowienczyk P, Alastruey J. *A database of virtual healthy subjects to assess the accuracy of foot-to-foot pulse wave velocities for estimation of aortic stiffness*. Am J Physiol Heart Circ Physiol, 2015. [DOI](https://doi.org/10.1152/ajpheart.00175.2015)

## Contact

**Primary Contact**: Dinh Bach Tue (tue.dinh@aalto.fi)  
**Advisor**: PhD Researcher Bahmani Nima

---

**Collaboration Welcome**: This work establishes a foundation for synthetic-to-real transfer learning in cardiovascular monitoring. We encourage contributions toward real-world wearable device deployment.
