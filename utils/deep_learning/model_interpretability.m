function model_interpretability(net, seqData, y_true, model_type)
%MODEL_INTERPRETABILITY Analyze which parts of input are most important
%
% Inputs:
% net - trained network (CNN or GRU)
% seqData - cell array of input sequences [2 x T] (PPG, Area)
% y_true - true PWV values
% model_type - 'CNN' or 'GRU' or 'features'

addpath('../utils/others');

fprintf('\n=== Model Interpretability Analysis ===\n');

% Select subset for analysis (computational efficiency)
n_samples = min(50, length(seqData));
idx = randperm(length(seqData), n_samples);
data_subset = seqData(idx);
y_subset = y_true(idx);

% Get baseline predictions
y_pred = predict(net, data_subset);

% 1. Occlusion Analysis
fprintf('Running Occlusion Analysis...\n');
occlusion_importance = occlusion_analysis(net, data_subset, y_pred);

% 2. Perturbation Analysis (SHAP-like) - COMMENTED OUT (takes 3+ hours)
fprintf('Running Perturbation Analysis...\n');
perturbation_importance = perturbation_analysis(net, data_subset, y_pred);
%perturbation_importance = zeros(2, size(data_subset{1}, 2)); % Dummy data for visualization

% Visualize results
visualize_importance(occlusion_importance, perturbation_importance, model_type);
save_figure(sprintf('interpretability_%s', lower(model_type)), 9);

% Save results
save(sprintf('interpretability_%s.mat', lower(model_type)), 'perturbation_importance', ...
'occlusion_importance'); % perturbation_importance commented out

end

function importance = occlusion_analysis(net, seqData, y_baseline)
%OCCLUSION_ANALYSIS Systematically mask parts of input and measure impact

n_samples = length(seqData);
T = size(seqData{1}, 2); % Time points
n_channels = size(seqData{1}, 1); % PPG + Area

% Window size for occlusion (10% of signal length)
window_size = max(5, round(0.1 * T));
n_windows = floor(T / window_size);

importance = zeros(n_channels, n_windows);

for ch = 1:n_channels
for w = 1:n_windows
% Create occluded data
occluded_data = seqData;
start_idx = (w-1) * window_size + 1;
end_idx = min(w * window_size, T);
for i = 1:n_samples
occluded_data{i}(ch, start_idx:end_idx) = 0; % Zero out window
end
% Get predictions with occlusion
y_occluded = predict(net, occluded_data);
% Importance = change in prediction accuracy
importance(ch, w) = mean(abs(y_baseline - y_occluded));
end
end

end

function importance = perturbation_analysis(net, seqData, y_baseline)
%PERTURBATION_ANALYSIS Add noise to different parts and measure sensitivity

n_samples = length(seqData);
T = size(seqData{1}, 2);
n_channels = size(seqData{1}, 1);

% Perturbation parameters
noise_levels=[0.15 0.3]; % Relative noise levels
n_perturbations = 10; % Monte Carlo samples

importance = zeros(n_channels, T);

for ch = 1:n_channels
for t=1:T
sensitivity_scores = [];
for noise_level = noise_levels
for rep = 1:n_perturbations
% Create perturbed data
perturbed_data = seqData;
for i = 1:n_samples
% Add Gaussian noise to specific time point and channel
signal_std = std(seqData{i}(ch, :));
noise = noise_level * signal_std * randn();
perturbed_data{i}(ch, t) = seqData{i}(ch, t) + noise;
end
% Get predictions with perturbation
y_perturbed = predict(net, perturbed_data);
% Sensitivity = change in prediction
sensitivity = mean(abs(y_baseline - y_perturbed));
sensitivity_scores(end+1) = sensitivity;
end
end
% Average sensitivity across noise levels and repetitions
importance(ch, t) = mean(sensitivity_scores);
end
end

end

function visualize_importance(occlusion_imp, perturbation_imp, model_type)
%VISUALIZE_IMPORTANCE Plot importance maps

figure('Position', [100, 100, 1200, 800]);

% Occlusion Analysis Results
subplot(2,2,1);
imagesc(occlusion_imp);
colorbar;
title(sprintf('%s: Occlusion Importance', model_type));
xlabel('Time Windows');
ylabel('Channel (1=PPG, 2=Area)');
yticks([1 2]);
yticklabels({'PPG', 'Area'});

subplot(2,2,2);
plot(mean(occlusion_imp, 1), 'LineWidth', 2);
title('Average Occlusion Importance');
xlabel('Time Windows');
ylabel('Importance Score');
grid on;

% Perturbation Analysis Results
subplot(2,2,3);
imagesc(perturbation_imp);
colorbar;
title(sprintf('%s: Perturbation Sensitivity', model_type));
xlabel('Time Points');
ylabel('Channel (1=PPG, 2=Area)');
yticks([1 2]);
yticklabels({'PPG', 'Area'});

subplot(2,2,4);
hold on;
plot(perturbation_imp(1,:), 'LineWidth', 2, 'DisplayName', 'PPG');
plot(perturbation_imp(2,:), 'LineWidth', 2, 'DisplayName', 'Area');
title('Perturbation Sensitivity by Channel');
xlabel('Time Points');
ylabel('Sensitivity Score');
legend;
grid on;

% Summary statistics
fprintf('\n=== Interpretability Summary ===\n');
[~, max_ch] = max(mean(occlusion_imp,2));
channel_names = {'PPG', 'Area'};
fprintf('Most important channel (occlusion): %s\n', channel_names{max_ch});
[~, max_win] = max(mean(occlusion_imp,1));
fprintf('Most important time region (occlusion): Window %d\n', max_win);

[~, max_pert_time] = max(mean(perturbation_imp,1));
fprintf('Most sensitive time point (perturbation): %d (%.1f%% of cycle)\n', ...
max_pert_time, 100*max_pert_time/size(perturbation_imp,2));

end