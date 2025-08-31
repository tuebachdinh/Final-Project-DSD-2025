%% PWDB Analysis - Complete Refactored Main Script
% Machine Learning Analysis of Wrist Pulse Waveforms for Arterial Stiffness Estimation
% All 9 parts organized into separate modules

clear; clc; close all;

fprintf('=== PWDB Complete Analysis Pipeline ===\n\n');

%% Part 1: Data Preparation and Wave Extraction
[waves, haemods, PWV_cf, age, fs, plaus_idx, data] = part1_data_preparation();

%% Part 2: Visualization
part2_visualization(waves, PWV_cf, age, fs);

%% Part 3: PTT Analysis
part3_ptt_analysis(data, plaus_idx, age);

%% Part 4: Feature-based Regression
part4_feature_regression(data, plaus_idx, PWV_cf, haemods);

%% Part 5: ML with Waveforms
part5_ml_waveforms(waves, PWV_cf, age, fs);

%% Part 6: Signal Processing
part6_signal_processing(waves, fs);

%% Part 7: Data Augmentation
[waves_augmented, PWV_cf_augmented] = part7_data_augmentation(waves, PWV_cf, fs);

%% Part 8: Classical ML with Augmented Data
part8_classical_ml(waves_augmented, PWV_cf_augmented, fs);

%% Part 9: Deep Learning
part9_deep_learning(waves, PWV_cf, waves_augmented, PWV_cf_augmented);

%% Part 10: Interpretability for models from part 9 
part10_model_interpretability()

fprintf('\n=== Complete Analysis Pipeline Finished ===\n');

%% Part 11: Summerize results and explain
part11_summarize_metrics()

%% Part 12: Transfer clean data to augmented model 
part12_check_best_model('../models/part9_models_augmented_both.mat', waves, PWV_cf, 'both',  fs);

% Output: == Transfer Eval on CLEAN == R^2 = 0.9391 | MAE = 0.3008 m/s | RMSE = 0.4531 m/s

