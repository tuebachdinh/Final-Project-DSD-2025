%% PWDB Analysis - Complete Refactored Main Script
% Machine Learning Analysis of Wrist Pulse Waveforms for Arterial Stiffness Estimation
% All 9 parts organized into separate modules

clear; clc; close all;
addpath('../utils/others');
addpath('../utils/deep_learning');

fprintf('=== PWDB Complete Analysis Pipeline ===\n\n');

%% Part 1: Data Preparation and Wave Extraction
[waves, haemods, PWV_cf, age, fs, plaus_idx, data] = part1_data_preparation();

%% Part 2: Visualization
part2_visualization(waves, PWV_cf, age, fs);

%% Part 3: PTT Analysis
part3_ptt_analysis(data, plaus_idx, age);

%% Part 4: Feature-based Regression
part4_feature_regression(data, plaus_idx, PWV_cf, haemods);

%% Part 5: ML with Waveforms and example waveforms of 1 subject
part5_ml_waveforms(waves, PWV_cf, age, fs);

%% Part 6: Signal Processing
part6_signal_processing(waves, fs);

%% Part 7: Data Augmentation
[waves_augmented, PWV_cf_augmented] = part7_data_augmentation(waves, PWV_cf, fs);

%% Part 8: Classical ML with Augmented Data
% part8_classical_ml(waves_augmented, PWV_cf_augmented, fs);
% Uncomment for computational heaviness, as we are using the signal processing
% function for each subject each waveform.

%% Part 9: Deep Learning
% We train 6 times with 6 different inputs: PPG, Area, PPG+Area, augmented
% PPG, augmented Area and augmented(PPG+Area). The input is feed into 3
% models (CNN, GRU, TCN), which results later in 18 models.

part9_deep_learning(waves, PWV_cf, waves_augmented, PWV_cf_augmented, 'clean', 'both');
part9_deep_learning(waves, PWV_cf, waves_augmented, PWV_cf_augmented, 'clean', 'ppg');
part9_deep_learning(waves, PWV_cf, waves_augmented, PWV_cf_augmented, 'clean', 'area');
 
part9_deep_learning(waves, PWV_cf, waves_augmented, PWV_cf_augmented, 'augmented', 'both');
part9_deep_learning(waves, PWV_cf, waves_augmented, PWV_cf_augmented, 'augmented', 'ppg');
part9_deep_learning(waves, PWV_cf, waves_augmented, PWV_cf_augmented, 'augmented', 'area');


%% Part 10: Interpretability for models from part 9 
% This results in 3 files: interpretability_augmented_tcn/gru/cnn.mat 
part10_model_interpretability(waves, PWV_cf, 'part9_models_augmented_both.mat');

% This results in 3 files: interpretability_tcn/gru/cnn.mat
part10_model_interpretability(waves, PWV_cf, 'part9_models_clean_both.mat');

% This results in 1 files: interpretability_final_tcn.mat
part10_model_interpretability(waves, PWV_cf, 'tcn_fold3.mat');

% NOTE!!!!: The intepretability_.mat files will be in src/, please manually
% move it intomodels/ folder for later analysis.

%% Summerize results and create tables (csv files in table folder)
summarize_metrics_create_table('clean');
summarize_metrics_create_table('augmented');

%% Check parameters of a model file (3 models in 1 file)
count_params('../models/part9_models_clean_both.mat');

%% Make stack plot based on tables.
make_stacked_plots('../tables/part9_table_clean_9x4.csv','../tables/part9_table_augmented_9x4.csv')

%% Overlay importance on sample subject
addpath('../utils/deep_learning');
sample_idx = 2675; % pick any subject row (1..3837)

plot_importance_overlay('part10_interpretability_final_tcn.mat', waves, sample_idx, 'tcn');

plot_importance_overlay('part10_interpretability_tcn.mat', waves, sample_idx, 'tcn');

plot_importance_overlay('part10_interpretability_cnn.mat', waves, sample_idx, 'cnn');

plot_importance_overlay('part10_interpretability_gru.mat', waves, sample_idx, 'gru');

plot_importance_overlay('part10_interpretability_augmented_tcn.mat', waves, sample_idx, 'tcn');

plot_importance_overlay('part10_interpretability_augmented_cnn.mat', waves, sample_idx, 'cnn');

plot_importance_overlay('part10_interpretability_augmented_gru.mat', waves, sample_idx, 'gru');


%% Plot model architecture and parametets
S = load(fullfile('..','models', 'part9_models_augmented_both.mat'));
analyzeNetwork(S.net_cnn);
analyzeNetwork(S.net_gru);
analyzeNetwork(S.net_tcn);

% More GUI way, drag-drop
% deepNetworkDesigner(S.net_tcn) 
% deepNetworkDesigner(S.net_tcn) 
% deepNetworkDesigner(S.net_tcn)  

%% Part 12: 5-fold cross validation on best model (TCN - both PPG + Area as inputs)
% Using both clean and augmented data after seeing the benefits of augmentation
% This function results in 5 models .mat and 1 metrics file part12_.../.mat
part12_cross_validation(waves, PWV_cf, waves_augmented, PWV_cf_augmented, 5, 'both')

% time=854.7s, time=739.5s, time=583.1s, time=563.0s, time=576.4s