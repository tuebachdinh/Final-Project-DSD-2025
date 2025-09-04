function part10_model_interpretability(waves, PWV_cf, file)
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
model_file = fullfile(models_dir, file);
if ~exist(model_file, 'file')
    error('Models not found. Run part9_deep_learning() first.');
end



if contains(file, 'augmented')
    load(model_file, 'net_cnn', 'net_gru', 'net_tcn');
    % Run interpretability analysis
    fprintf('\n--- Running CNN Interpretability Analysis ---\n');
    model_interpretability(net_cnn, waves, PWV_cf, 'augmented_cnn');
    fprintf('\n--- Running GRU Interpretability Analysis ---\n');
    model_interpretability(net_gru, waves, PWV_cf, 'augmented_gru');
    fprintf('\n--- Running TCN Interpretability Analysis ---\n');
    model_interpretability(net_tcn, waves, PWV_cf, 'augmented_tcn');
elseif contains(file, 'clean')
    load(model_file, 'net_cnn', 'net_gru', 'net_tcn');
    % Run interpretability analysis
    fprintf('\n--- Running CNN Interpretability Analysis ---\n');
    model_interpretability(net_cnn, waves, PWV_cf, 'CNN');
    fprintf('\n--- Running GRU Interpretability Analysis ---\n');
    model_interpretability(net_gru, waves, PWV_cf, 'GRU');
    fprintf('\n--- Running TCN Interpretability Analysis ---\n');
    model_interpretability(net_tcn, waves, PWV_cf, 'TCN');
else
    load(model_file, 'net_tcn');
    model_interpretability(net_tcn, waves, PWV_cf, 'final_tcn');
end



fprintf('Part 10: Model interpretability completed\n');
end