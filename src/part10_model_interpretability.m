function part10_model_interpretability()
%PART10_MODEL_INTERPRETABILITY Model Interpretability Analysis
addpath('../utils/others');
addpath('../utils/deep_learning');

fprintf('\n=== Part 10: Model Interpretability Analysis ===\n');

% Find project root and load models
current_dir = pwd;
while ~exist(fullfile(current_dir, 'literature-review.md'), 'file')
    parent_dir = fileparts(current_dir);
    if strcmp(current_dir, parent_dir)
        error('Could not find project root');
    end
    current_dir = parent_dir;
end
models_dir = fullfile(current_dir, 'models');

% Load trained models and test data
model_file = fullfile(models_dir, 'part9_models_augmented_both.mat');
if ~exist(model_file, 'file')
    error('Models not found. Run part9_deep_learning() first.');
end

load(model_file, 'net_cnn', 'net_gru', 'net_tcn', 'test_data');

fprintf('Loaded models and test data from Part 9\n');
fprintf('Test samples: %d\n', length(test_data.seqData));

% Run interpretability analysis
% fprintf('\n--- Running CNN Interpretability Analysis ---\n');
% model_interpretability(net_cnn, test_data.seqData, test_data.ytrue, 'CNN');
% 
% fprintf('\n--- Running GRU Interpretability Analysis ---\n');
% model_interpretability(net_gru, test_data.seqData, test_data.ytrue, 'GRU');

fprintf('\n--- Running TCN Interpretability Analysis ---\n');
model_interpretability(net_tcn, test_data.seqData, test_data.ytrue, 'TCN');


fprintf('Part 10: Model interpretability completed\n');
end