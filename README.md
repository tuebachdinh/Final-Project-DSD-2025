# Deep Learning Pipeline for Arterial Stiffness Estimation from Wrist Pulse Signals

## Project Goal

This project develops a **comprehensive deep learning pipeline** for predicting arterial stiffness from wrist pulse waveforms using **advanced signal processing, data visualization, classical machine learning models, deep neural networks (CNN/GRU), and model interpretability techniques**. The pipeline progresses from in silico data analysis to real-world device deployment through **transfer learning and domain adaptation**.

Our objective is to create a **robust, interpretable, and deployable system** for non-invasive cardiovascular health monitoring using consumer wearable devices.

---

## Motivation

- **Clinical Impact:** Arterial stiffness (cfPWV) predicts cardiovascular events but requires expensive clinical equipment
- **Wearable Revolution:** Consumer devices can capture pulse signals but lack validated algorithms for health assessment
- **AI Opportunity:** Advanced ML/DL can extract physiological insights from noisy, real-world wearable data
- **Interpretability Need:** Clinical adoption requires understanding what models learn from pulse morphology
- **Transfer Challenge:** Bridging the gap between clean simulated data and noisy real-world signals

---

## Datasets

### 1. PWDB In Silico Database (Charlton et al., Am J Physiol, 2019)(main dataset)
- **4,374 virtual subjects** with simulated pressure, flow, area, and PPG waveforms at multiple sites (including wrist PPG).
- Includes demographic and hemodynamic diversity (ages 25–75).
- Download: [https://github.com/peterhcharlton/pwdb](https://github.com/peterhcharlton/pwdb)
- Reference:  
  Charlton PH, Mariscal Harana J, Vennin S, Li Y, Chowienczyk P, Alastruey J.  
  *Modeling arterial pulse waves in healthy aging: a database for in silico evaluation of hemodynamics and pulse wave indexes*.  
  Am J Physiol Heart Circ Physiol, 2019. [Link](https://doi.org/10.1152/ajpheart.00218.2019)

### 2. HaeMod Virtual Subjects (Willemet et al., Am J Physiol, 2015)(archieved - no PPG)
- **3,325 virtual subjects** with simulated pressure, flow, and area waveforms at multiple arterial locations, including the wrist (radial artery).
- Provided simulation parameters and computed indices (cfPWV, cardiac output, geometry).
- Download and documentation: [http://haemod.uk/original](http://haemod.uk/original)
- Reference:  
  Willemet M, Chowienczyk P, Alastruey J.  
  *A database of virtual healthy subjects to assess the accuracy of foot-to-foot pulse wave velocities for estimation of aortic stiffness*.  
  Am J Physiol Heart Circ Physiol, 2015. [Link](https://doi.org/10.1152/ajpheart.00175.2015)

---

## Repository Structure

```
Final-Project-DSD-2025/
├── src/                      # Analysis pipeline (Parts 1-9)
├── utils/                    # Utility functions
│   ├── others/              # General utilities (save_figure)
│   ├── deep_learning/       # Model interpretability
│   └── signal_processing/   # PPG/pulse analysis
├── data/                    # Dataset storage
├── models/                  # Trained models
├── images/                  # Generated figures
├── archieved           # Archieved work from original dataset 
└── README.md               # This file
```

## Analysis Workflow Pipeline

### Part 1: Data Preparation (`src/1_data_preparation.m`)
- Load PWDB dataset and filter plausible subjects
- Extract waveforms (pressure, flow, area, PPG) at multiple sites
- Prepare data structures for analysis

### Part 2: Visualization (`src/2_visualization.m`)
- Plot waveforms by PWV groups (low/medium/high stiffness)
- Visualize age-related changes in pulse morphology
- Generate comparative plots across signal types

### Part 3: PTT Analysis (`src/3_ptt_analysis.m`)
- Analyze pulse transit time from heart to wrist
- Examine PTT distribution and age relationships
- Validate timing-based stiffness indicators

### Part 4: Feature Regression (`src/4_feature_regression.m`)
- Linear regression with area-based features
- Tree regression with classical PPG indices (RI, SI, AGI)
- Compare feature-based vs. direct waveform approaches

### Part 5: ML Waveforms (`src/5_ml_waveforms.m`)
- Machine learning on raw waveform time series
- Regression trees for area and PPG signals
- Performance comparison across signal modalities

### Part 6: Signal Processing (`src/6_signal_processing.m`)
- Advanced PPG feature extraction algorithms
- Dicrotic notch detection and Gaussian fitting
- Signal quality assessment and morphology analysis

### Part 7: Data Augmentation (`src/7_data_augmentation.m`)
- Real-world noise simulation (baseline drift, motion artifacts)
- SNR-controlled augmentation for robustness testing
- Clean vs. augmented signal comparison

### Part 8: Classical ML (`src/8_classical_ml.m`)
- Ridge regression and decision trees on augmented data
- Comprehensive feature extraction from PPG and area signals
- Model comparison and performance evaluation

### Part 9: Deep Learning (`src/9_deep_learning.m`)
- CNN and GRU models for waveform-to-PWV prediction
- Attention mechanisms for interpretability
- Model interpretability analysis (occlusion, perturbation, attention)

## Key Features

### Model Interpretability
- **Occlusion Analysis**: Identify critical waveform regions
- **Perturbation Analysis**: Measure sensitivity to noise
- **Attention Mechanisms**: Visualize model focus areas
- **Feature Importance**: Rank predictive features

### Real-World Simulation
- Noise injection (25 dB SNR)
- Baseline drift simulation
- Motion artifact modeling
- Signal quality assessment

---

## Pipeline Progression

### Phase 1: In Silico Foundation (Current)
- **Data Visualization**: Comprehensive waveform analysis by stiffness and age groups
- **Signal Processing**: Advanced PPG feature extraction and morphology analysis
- **Classical ML**: Feature-based regression with interpretability
- **Deep Learning**: CNN/GRU models with attention mechanisms
- **Model Interpretability**: Occlusion, perturbation, and attention analysis
- **Data Augmentation**: Real-world noise simulation for robustness

### Phase 2: Real-World Deployment (Future)
- **Transfer Learning**: Adapt in silico models to real device data
- **Domain Adaptation**: Bridge simulation-to-reality gap
- **Device Calibration**: Account for sensor-specific characteristics
- **Personalization**: Individual-specific model fine-tuning
- **Clinical Validation**: Validate against gold-standard measurements

## Technical Innovations

- **Multi-Modal Analysis**: Combines pressure, flow, area, and PPG signals
- **Interpretable AI**: Explains which waveform features drive predictions
- **Robust Augmentation**: Simulates real-world noise, drift, and motion artifacts
- **Attention Mechanisms**: Identifies critical temporal regions in pulse cycles
- **End-to-End Pipeline**: From raw signals to clinical predictions

---

## How to Use

### Prerequisites
- MATLAB R2020b or later with Deep Learning Toolbox
- PWDB dataset (download from [GitHub](https://github.com/peterhcharlton/pwdb))

### Quick Start
1. **Setup Data**: Place `pwdb_data.mat` in `data/exported_data/`
2. **Run Analysis**: Execute `src/main.m` or individual parts
3. **View Results**: Check `images/` for generated plots and `models/` for trained networks

### Running Individual Parts
```matlab
cd src/
[waves, haemods, PWV_cf, age, fs, plaus_idx, data] = part1_data_preparation();
part2_visualization(waves, PWV_cf, age, fs);
% ... continue with other parts
```

### Output Structure
- **Figures**: Automatically saved to `images/part[N]_[description].png`
- **Models**: Trained networks saved to `models/part9_cnn_gru_models.mat`
- **Analysis**: Interpretability results in `.mat` files

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
We encourage contributions to advance the pipeline from simulation to real-world deployment, bridging AI research with clinical cardiovascular monitoring.
