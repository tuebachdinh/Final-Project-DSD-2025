# Models Directory

This directory contains trained deep learning models and analysis results from the PWDB arterial stiffness estimation pipeline.

## File Structure

### Deep Learning Models (Part 9)
- **`part9_models_clean_both.mat`** - CNN/GRU/TCN trained on clean data (PPG+Area channels)
- **`part9_models_clean_PPG.mat`** - CNN/GRU/TCN trained on clean data (PPG only)
- **`part9_models_clean_Area.mat`** - CNN/GRU/TCN trained on clean data (Area only)
- **`part9_models_augmented_both.mat`** - CNN/GRU/TCN trained on augmented data (PPG+Area channels)
- **`part9_models_augmented_PPG.mat`** - CNN/GRU/TCN trained on augmented data (PPG only)
- **`part9_models_augmented_Area.mat`** - CNN/GRU/TCN trained on augmented data (Area only)

### Model Interpretability Results (Part 10)
- **`part10_interpretability_cnn.mat`** - CNN interpretability analysis (clean data)
- **`part10_interpretability_gru.mat`** - GRU interpretability analysis (clean data)
- **`part10_interpretability_tcn.mat`** - TCN interpretability analysis (clean data)
- **`part10_interpretability_augmented_cnn.mat`** - CNN interpretability analysis (augmented data)
- **`part10_interpretability_augmented_gru.mat`** - GRU interpretability analysis (augmented data)
- **`part10_interpretability_augmented_tcn.mat`** - TCN interpretability analysis (augmented data)
- **`part10_interpretability_final_tcn.mat`** - TCN interpretability from cross-validation fold

### Cross-Validation Models (Part 12)
- **`tcn_fold1.mat`** to **`tcn_fold5.mat`** - Individual TCN models from 5-fold cross-validation
- **`part12_cv_metrics.mat`** - Cross-validation performance metrics and summary

## MAT File Contents

### Part 9 Model Files (`part9_models_*.mat`)
```matlab
% Variables contained:
net_cnn         % Trained CNN model (SeriesNetwork/DAGNetwork)
net_gru         % Trained GRU model (SeriesNetwork/DAGNetwork)  
net_tcn         % Trained TCN model (DAGNetwork)
best_model      % String indicating best performing model
best_net        % Best performing network object
metrics         % Structure with performance metrics:
                %   .CNN: R2, MAE, RMSE, training_time
                %   .GRU: R2, MAE, RMSE, training_time
                %   .TCN: R2, MAE, RMSE, training_time
test_data       % Structure with test data:
                %   .seqData: Cell array of test sequences
                %   .ytrue: True PWV values for test set
```

### Part 10 Interpretability Files (`part10_interpretability_*.mat`)
```matlab
% Variables contained:
occlusion_importance      % Matrix [channels × windows] - Occlusion analysis results
perturbation_importance   % Matrix [channels × timepoints] - Perturbation sensitivity
```

### Cross-Validation Fold Files (`tcn_fold*.mat`)
```matlab
% Variables contained:
net_tcn         % Trained TCN model for this fold
```

### Cross-Validation Metrics (`part12_cv_metrics.mat`)
```matlab
% Variables contained:
results_summary % Structure with:
                %   .k: Number of folds
                %   .which_channels: Input channels used
                %   .per_fold: Metrics for each fold
                %   .summary: Aggregated statistics (mean ± std)
```

## Model Specifications

### CNN Architecture
- Input: [channels × time] sequences
- Conv1D(kernel=10, filters=48) → ReLU
- Conv1D(kernel=10, filters=96) → ReLU  
- GlobalAveragePooling1D
- FC(48) → ReLU → FC(1)
- Parameters: ~60k

### GRU Architecture
- Input: [channels × time] sequences
- GRU(96, return_sequences=True)
- Dropout(0.15)
- GRU(48, return_sequences=False)
- FC(48) → ReLU → FC(1)
- Parameters: ~60k

### TCN Architecture
- Input: [channels × time] sequences
- 2 Residual blocks with dilations [1,2] and [4,8]
- Each block: Conv1D(5,F) → ReLU → Dropout(0.1) → Conv1D(5,F) → ReLU + Skip
- GlobalAveragePooling1D → FC(48) → ReLU → FC(1)
- Parameters: ~65k (F=55)

## Usage Examples

### Load and Use Models
```matlab
% Load trained models
load('part9_models_clean_both.mat');

% Use best model for prediction
predictions = predict(best_net, test_sequences);

% Check model performance
fprintf('Best model: %s (R² = %.3f)\n', best_model, metrics.(best_model).R2);
```

### Load Interpretability Results
```matlab
% Load interpretability analysis
load('part10_interpretability_tcn.mat');

% Visualize importance
imagesc(occlusion_importance);
title('Occlusion Importance by Channel and Time Window');
```

### Load Cross-Validation Results
```matlab
% Load CV summary
load('part12_cv_metrics.mat');

% Display results
fprintf('5-fold CV Results:\n');
fprintf('Clean: R² = %.3f ± %.3f\n', ...
    results_summary.summary.clean.R2_mean, ...
    results_summary.summary.clean.R2_std);
```

