# PWDB Analysis - Complete Pipeline Documentation

This comprehensive pipeline analyzes wrist pulse waveforms for arterial stiffness estimation using advanced signal processing, classical ML, and deep learning approaches. The analysis progresses from data preparation through deep learning model interpretability.

## File Structure

### Main Scripts
- **`main.m`** - Complete pipeline running all parts with detailed workflow

### Analysis Pipeline (12 Parts)
1. **`part1_data_preparation.m`** - Data loading and wave extraction
2. **`part2_visualization.m`** - Waveform visualization by PWV/age groups  
3. **`part3_ptt_analysis.m`** - Pulse transit time analysis
4. **`part4_feature_regression.m`** - Linear and tree regression with features
5. **`part5_ml_waveforms.m`** - ML using time series on full waveforms
6. **`part6_signal_processing.m`** - Advanced signal processing for feature extraction
7. **`part7_data_augmentation.m`** - Real-world noise simulation and data augmentation
8. **`part8_classical_ml.m`** - Classical ML with comprehensive feature extraction
9. **`part9_deep_learning.m`** - Deep learning models (CNN/GRU/TCN) with multiple configurations
10. **`part10_model_interpretability.m`** - Model interpretability analysis
11. **`part11_transfer_eval.m`** - Currently not commited as no usage
12. **`part12_cross_validation.m`** - K-fold cross-validation for robust evaluation

## Usage

### Run Complete Analysis
```matlab
main  % Runs entire pipeline with all parts
```

### Run Individual Parts
It is recommended to run each part in defined order. Each part can be runned by clicking the block on the left side bar in Matlab. Remember to open the src/ folder before running or multiple functions will not work. Part 1 and part 7 are compulsory to run as they provide variables in the workspace.
```matlab
% Part 1: Data preparation
[waves, haemods, PWV_cf, age, fs, plaus_idx, data] = part1_data_preparation();

%% Part 7: Data Augmentation
[waves_augmented, PWV_cf_augmented] = part7_data_augmentation(waves, PWV_cf, fs);

% Part 9: Deep learning with specific configuration
part9_deep_learning(waves, PWV_cf, waves_augmented, PWV_cf_augmented, 'clean', 'both');
```

## Detailed Part Descriptions

### Part 1: Data Preparation and Wave Extraction
**Function**: `part1_data_preparation()`

**Purpose**: Loads and preprocesses PWDB dataset for analysis

**Implementation**:
- Loads PWDB data from `../data/exported_data/pwdb_data.mat`
- Filters for physiologically plausible subjects using `data.plausibility.plausibility_log`
- Extracts waveforms for 4 signal types (P, U, A, PPG) at 5 anatomical sites
- Truncates all waveforms to minimum length for consistent analysis
- Processes 4,374 virtual subjects → ~3,837 plausible subjects

**Outputs**:
- `waves`: Structure with truncated waveform matrices (subjects × time)
- `haemods`: Hemodynamic parameters for each subject
- `PWV_cf`: Carotid-femoral pulse wave velocity (target variable)
- `age`: Subject ages, `fs`: Sampling frequency (500 Hz)

**Helper Functions**: None (standalone)

---

### Part 2: Visualization
**Function**: `part2_visualization(waves, PWV_cf, age, fs)`

**Purpose**: Creates comprehensive visualizations of waveform patterns

**Implementation**:
- **PWV Group Analysis**: Divides subjects into tertiles (low/medium/high PWV)
- **Age Group Analysis**: Compares waveforms across age groups (25-75 years)
- **Statistical Visualization**: Plots mean ± 0.5σ confidence bands
- **Multi-signal Display**: Shows P, U, A, PPG waveforms simultaneously

**Key Features**:
- Quantile-based PWV grouping with automatic labeling
- Shaded confidence bands with boundary lines
- Consistent color schemes across plots
- Automatic axis scaling and grid formatting

**Helper Functions**:
- `save_figure()` from `utils/others/` - Saves figures with consistent naming

---

### Part 3: PTT Analysis  
**Function**: `part3_ptt_analysis(data, plaus_idx, age)`

**Purpose**: Analyzes pulse transit time from heart to wrist

**Implementation**:
- Calculates PTT using onset times: `PTT = onset_radial - onset_aortic`
- Uses both direct onset times and pulse wave indices
- Filters positive PTT values for physiological validity
- Creates distribution histograms and age-relationship boxplots

**Key Metrics**:
- Mean PTT: ~0.1-0.2 seconds (heart to wrist)
- Age correlation analysis via boxplots
- Statistical summaries with standard deviations

**Helper Functions**:
- `save_figure()` from `utils/others/` - Figure saving with part numbering

---

### Part 4: Feature-based Regression
**Function**: `part4_feature_regression(data, plaus_idx, PWV_cf, haemods)`

**Purpose**: Classical regression using engineered features

**Implementation**:
- **Linear Regression**: Uses area-based features (Age, Amax, Amean, Amin)
- **Tree Regression**: Uses classical PPG indices (RI, SI, AGI_mod)
- **Train/Test Split**: 80/20 split with cross-validation
- **Performance Metrics**: Correlation coefficients and residual analysis

**Features Used**:
- **Area Features**: Maximum, mean, minimum luminal area
- **PPG Indices**: Reflection Index, Stiffness Index, Augmentation Index

**Helper Functions**:
- `save_figure()` from `utils/others/` - Residual and prediction plots

---

### Part 5: ML with Waveforms
**Function**: `part5_ml_waveforms(waves, PWV_cf, age, fs)`

**Purpose**: Machine learning on raw time-series waveforms

**Implementation**:
- **Area Waveform ML**: Regression trees on full area time series
- **PPG Waveform ML**: Regression trees on full PPG time series  
- **Direct Waveform Input**: No feature engineering, raw temporal data
- **Performance Comparison**: Correlation analysis between approaches

**Key Innovation**: Uses entire waveform as input (high-dimensional) rather than extracted features

**Visualization**: Example subject waveforms across all signal types

**Helper Functions**:
- `save_figure()` from `utils/others/` - Regression plots and example waveforms

---

### Part 6: Signal Processing
**Function**: `part6_signal_processing(waves, fs)`

**Purpose**: Advanced PPG signal analysis with multiple algorithms

**Implementation**:
- **PulseAnalyse10**: Comprehensive fiducial point detection
- **Gaussian Fitting**: Two-Gaussian model for P1/P2 peak detection
- **Feature Extraction**: Frequency and morphology features
- **Signal Quality**: Template-based quality assessment
- **New PulseAnalyse**: Enhanced analysis with quality metrics

**Advanced Features**:
- Dicrotic notch detection with visualization
- Harmonic analysis and spectral features
- Morphological parameters (skewness, kurtosis, area under curve)
- Multi-algorithm comparison for robustness

**Helper Functions**:
- `PulseAnalyse10()` from `utils/signal_processing/` - Fiducial point detection
- `fitTwoGaussiansPPG()` from `utils/signal_processing/` - Gaussian peak fitting
- `extractFreqMorphFeatures()` from `utils/signal_processing/` - Feature extraction
- `computeSimpleSQI()` from `utils/signal_processing/` - Signal quality index
- `PulseAnalyse()` from `utils/signal_processing/` - Enhanced pulse analysis
- `save_figure()` from `utils/others/` - Signal processing visualizations

---

### Part 7: Data Augmentation
**Function**: `part7_data_augmentation(waves, PWV_cf, fs)`

**Purpose**: Simulates real-world signal degradation for robustness testing

**Implementation**:
- **Noise Injection**: Gaussian noise for ~25 dB SNR
- **Baseline Drift**: Sinusoidal drift (0.1-0.4 Hz, 5% amplitude for PPG, 2% for Area)
- **Motion Artifacts**: Random spikes (30% probability, exponential decay)
- **Signal-Specific**: Different augmentation rate for PPG vs Area

**Quality Control**:
- SNR measurement and reporting
- Visual comparison plots (clean vs augmented)
- Maintains physiological plausibility

**Innovation**: Realistic artifact simulation based on wearable device characteristics

**Helper Functions**:
- `save_figure()` from `utils/others/` - Augmentation comparison plots

---

### Part 8: Classical ML with Augmented Data
**Function**: `part8_classical_ml(waves_augmented, PWV_cf_augmented, fs)`

**Purpose**: Comprehensive classical ML with feature engineering on augmented data

**Implementation**:
- **Feature Extraction**: PPG indices (RI, SI, AGI_mod) + Area features (max, min, mean)
- **Model Training**: Ridge regression and decision trees
- **Robust Evaluation**: Z-score normalization and cross-validation
- **Performance Metrics**: R², MAE, RMSE with statistical comparison

**Advanced Processing**:
- Per-subject feature extraction with error handling
- Feature standardization for fair model comparison
- Best model selection based on R² performance

**Helper Functions**:
- `PulseAnalyse()` from `utils/signal_processing/` - Feature extraction per subject
- `save_figure()` from `utils/others/` - Model performance visualization

---

### Part 9: Deep Learning
**Function**: `part9_deep_learning(waves, PWV_cf, waves_augmented, PWV_cf_augmented, train_data_tag, which_channels)`

**Purpose**: Advanced deep learning with multiple architectures and configurations

**Implementation**:
- **Three Architectures**: CNN (1D convolutions), GRU (recurrent), TCN (temporal convolutional)
- **Multiple Configurations**: 6 training scenarios (clean/augmented × PPG/Area/both)
- **Channel Flexibility**: 1-channel (PPG or Area) or 2-channel (both) input
- **Robust Training**: Always validates on clean data regardless of training data

**Architecture Details**:
- **CNN**: Conv1D(10,48) → Conv1D(10,96) → GlobalAvgPool → FC(48) → FC(1)
- **GRU**: GRU(96,seq) → Dropout(0.15) → GRU(48,last) → FC(48) → FC(1)  
- **TCN**: 2 residual blocks with dilations (1,2,4,8) → GAP → FC(48) → FC(1)

**Advanced Features**:
- Per-sequence z-normalization
- Model reuse with preloading capability
- Comprehensive performance comparison
- Automatic best model selection

**Helper Functions**:
- `save_figure()` from `utils/others/` - Model comparison plots
- Custom TCN architecture builder with residual connections

---

### Part 10: Model Interpretability
**Function**: `part10_model_interpretability(waves, PWV_cf, file)`

**Purpose**: Explains model decisions through interpretability analysis

**Implementation**:
- **Occlusion Analysis**: Systematically masks input regions to measure importance
- **Perturbation Analysis**: SHAP-like sensitivity analysis with noise injection
- **Multi-Model Support**: Works with CNN, GRU, and TCN architectures
- **Visualization**: Heatmaps and importance plots for temporal analysis

**Technical Details**:
- Window-based occlusion with configurable window sizes
- Monte Carlo perturbation with multiple noise levels
- Channel-wise importance analysis for multi-channel inputs
- Statistical aggregation across multiple samples

**Helper Functions**:
- `model_interpretability()` from `utils/deep_learning/` - Core interpretability engine
- `plot_importance_overlay()` from `utils/deep_learning/` - Importance visualization
- `save_figure()` from `utils/others/` - Interpretability plots

---

### Part 12: Cross-Validation. Final best model with both clean and augmented input
**Function**: `part12_cross_validation(waves, PWV_cf, waves_augmented, PWV_cf_augmented, k, which_channels)`

**Purpose**: Robust model evaluation using k-fold cross-validation. Training with both clean and augmented data

**Implementation**:
- **K-Fold Strategy**: Subject-wise splitting (default k=5)
- **Mixed Training**: Combines clean and augmented data for training
- **Dual Evaluation**: Tests on both clean and augmented versions of held-out subjects
- **TCN Focus**: Uses best-performing architecture from Part 9

**Statistical Rigor**:
- Per-fold metrics with mean ± std reporting
- Training time analysis
- Model persistence for each fold
- Comprehensive results summary

**Helper Functions**:
- Custom TCN architecture builder (replicated from Part 9)
- `save_figure()` from `utils/others/` - Cross-validation results


## Key Innovations

1. **Multi-Modal Analysis**: Combines pressure, flow, area, and PPG signals
2. **Realistic Augmentation**: Simulates wearable device artifacts and noise
3. **Architecture Comparison**: CNN vs GRU vs TCN with systematic evaluation
4. **Interpretability Focus**: Explains model decisions through multiple methods
5. **Robust Evaluation**: K-fold CV with mixed clean/augmented training
6. **End-to-End Pipeline**: From raw signals to interpretable predictions

## Pipeline Benefits

1. **Comprehensive Analysis**: Complete workflow from data to interpretable models
2. **Methodological Rigor**: Multiple validation approaches and robust evaluation
3. **Clinical Relevance**: Focus on non-invasive cardiovascular assessment
4. **Reproducibility**: Standardized functions with consistent interfaces
5. **Scalability**: Modular design supports extension and modification
6. **Interpretability**: Built-in model explanation capabilities
7. **Real-World Ready**: Augmentation simulates deployment conditions

## Dependencies

- MATLAB R2020b+ with Deep Learning Toolbox
- Signal Processing Toolbox  
- Statistics and Machine Learning Toolbox
- Custom utilities in `utils/` directory
- PWDB dataset in `data/exported_data/pwdb_data.mat`