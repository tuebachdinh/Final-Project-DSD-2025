# PWDB Analysis - Complete Refactored Structure

This refactored version organizes the original monolithic `main.m` script into 9 modular components for better maintainability, reusability, and understanding.

## File Structure

### Main Scripts
- **`main_complete.m`** - Complete pipeline running all 9 parts
- **`main.m`** - Original monolithic script (preserved for reference)

### Modular Components
1. **`part1_data_preparation.m`** - Data loading and wave extraction
2. **`part2_visualization.m`** - Waveform visualization by PWV/age groups
3. **`part3_ptt_analysis.m`** - Pulse transit time analysis
4. **`part4_feature_regression.m`** - Linear and tree regression with features
5. **`part5_ml_waveforms.m`** - ML using time series on full waveforms
6. **`part6_signal_processing.m`** - Signal processing algorithms for feature extraction
7. **`part7_data_augmentation.m`** - Data augmentation for real-world simulation
8. **`part8_classical_ml.m`** - Classical ML with augmented data
9. **`part9_deep_learning.m`** - Deep learning comparison (CNN vs GRU)

## Usage

### Run Complete Analysis
```matlab
main_complete
```

### Run Individual Parts
```matlab
% Part 1: Data preparation
[waves, haemods, PWV_cf, age, fs, plaus_idx, data] = part1_data_preparation();

% Part 2: Visualization
part2_visualization(waves, PWV_cf, age, fs);

% Part 3: PTT analysis
part3_ptt_analysis(data, plaus_idx, age);

% And so on...
```

## Part Descriptions

### Part 1: Data Preparation and Wave Extraction
- Loads PWDB data from MAT file
- Filters for physiologically plausible subjects
- Extracts and truncates waveforms for all wave types and sites
- Returns processed data structures

### Part 2: Visualization
- Creates PWV group comparison plots with confidence bands
- Generates age group analysis plots
- Handles all figure formatting and styling

### Part 3: PTT Analysis
- Calculates pulse transit times using onset times
- Creates PTT distribution histograms
- Generates PTT vs age boxplots

### Part 4: Feature-based Regression
- Performs linear regression with area features
- Implements tree regression with classical PPG indices
- Generates prediction vs actual plots

### Part 5: ML with Waveforms
- Uses full Area waveforms for ML prediction
- Uses full PPG waveforms for ML prediction
- Compares tree regression performance

### Part 6: Signal Processing
- Applies PulseAnalyse10 for fiducial point detection
- Performs Gaussian fitting for P1/P2 detection
- Extracts frequency and morphology features
- Computes signal quality indices

### Part 7: Data Augmentation
- Simulates realistic noise and artifacts
- Adds baseline drift and motion artifacts
- Creates augmented datasets for PPG and Area signals
- Visualizes clean vs augmented data

### Part 8: Classical ML with Augmented Data
- Extracts comprehensive features from augmented data
- Trains multiple classical ML models (Ridge, Tree, etc.)
- Compares model performance
- Saves best model for deployment

### Part 9: Deep Learning
- Implements CNN and GRU models for sequence analysis
- Uses 2-channel input (PPG + Area)
- Compares deep learning vs classical approaches
- Saves trained models

## Benefits of Refactoring

1. **Modularity** - Each analysis component is in its own file
2. **Reusability** - Functions can be called independently
3. **Maintainability** - Easier to modify specific parts
4. **Testing** - Individual components can be tested separately
5. **Collaboration** - Team members can work on different modules
6. **Understanding** - Clear separation of analysis steps
7. **Debugging** - Easier to isolate and fix issues

## Dependencies

- MATLAB Deep Learning Toolbox (for Part 9)
- Signal Processing Toolbox
- Statistics and Machine Learning Toolbox
- Custom algorithms in `/algorithms/` directory

## Output Files

- `part7_best_model.mat` - Best classical ML model from Part 8
- `part9_cnn_gru_models.mat` - Deep learning models from Part 9