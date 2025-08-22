function part9_deep_learning(waves_augmented, PWV_cf_augmented)
%PART9_DEEP_LEARNING Deep Learning Comparison (CNN vs GRU)
addpath('../utils/others');
addpath('../utils/deep_learning');

rng(8);

X_ppg = waves_augmented.PPG_Radial; % [N x T] - augmented data only
X_area = waves_augmented.A_Radial; % [N x T] - augmented data only
y = PWV_cf_augmented(:);

fprintf('\n=== Part 9: Deep Learning with Augmented Data ===\n');
fprintf('Training with %d subjects (augmented data only)\n', length(y));

good = all(isfinite(X_ppg),2) & all(isfinite(X_area),2) & isfinite(y);
X_ppg = X_ppg(good,:); 
X_area = X_area(good,:);
y = y(good);

N = size(X_ppg,1);
T = size(X_ppg,2);

% Normalize each signal (PPG already normalized, Area needs normalization)
X_area = (X_area - mean(X_area,2)) ./ max(std(X_area,[],2), eps);

X_combined = cat(3, X_ppg, X_area);  % [N x T x 2]
X_combined = permute(X_combined, [3 2 1]); % [2 x T x N] for MATLAB

seqData = cell(N,1);
for i = 1:N
    seqData{i} = X_combined(:,:,i); %[2 x T]
end

% Train/Test split
idx = randperm(N);
nTrain = round(0.6*N);
nVal = round(0.2*N);

valIdx = idx(1:nVal);
trainIdx = idx(nVal+1:nVal+nTrain);
testIdx = idx(nVal+nTrain+1:end);

% Optimization
opts = trainingOptions('adam', ...
    'InitialLearnRate', 1e-3, ...
    'MaxEpochs', 55, ...
    'MiniBatchSize', 32, ...
    'Shuffle', 'every-epoch', ...
    'Verbose', true, ...
    'Plots', 'training-progress', ...
    'ValidationData', {seqData(valIdx), y(valIdx)}, ...
    'ValidationFrequency', 20, ...
    'ValidationPatience', 15);

% CNN Model
fprintf('\n--- Training CNN Model ---\n');
tic;
layers = [
    sequenceInputLayer(2, 'MinLength', T, 'Name', 'input')
    convolution1dLayer(10, 64, 'Padding', 'same')
    reluLayer
    convolution1dLayer(10, 128, 'Padding', 'same')
    reluLayer
    globalAveragePooling1dLayer
    fullyConnectedLayer(64)
    reluLayer
    fullyConnectedLayer(1)
    regressionLayer
];

net_cnn = trainNetwork(seqData(trainIdx), y(trainIdx), layers, opts);
cnn_time = toc;

yp_cnn = predict(net_cnn, seqData(testIdx));
ytrue = y(testIdx);

resid_cnn = ytrue - yp_cnn;
R2_cnn = 1 - sum(resid_cnn.^2) / sum((ytrue - mean(ytrue)).^2);
MAE_cnn = mean(abs(resid_cnn));
RMSE_cnn = sqrt(mean(resid_cnn.^2));

% GRU Model
fprintf('\n--- Training GRU Model ---\n');
tic;
layers_gru = [
    sequenceInputLayer(2, 'Name', 'input')
    gruLayer(64, 'OutputMode', 'last')
    fullyConnectedLayer(32)
    reluLayer
    fullyConnectedLayer(1)
    regressionLayer
];

net_gru = trainNetwork(seqData(trainIdx), y(trainIdx), layers_gru, opts);
gru_time = toc;

yp_gru = predict(net_gru, seqData(testIdx));
resid_gru = ytrue - yp_gru;
R2_gru = 1 - sum(resid_gru.^2) / sum((ytrue - mean(ytrue)).^2);
MAE_gru = mean(abs(resid_gru));
RMSE_gru = sqrt(mean(resid_gru.^2));

fprintf('\n=== Part 9 Deep Learning Results ===\n');
fprintf('CNN:  R^2 = %.3f | MAE = %.3f | RMSE = %.3f\n', R2_cnn, MAE_cnn, RMSE_cnn);
fprintf('GRU:  R^2 = %.3f | MAE = %.3f | RMSE = %.3f\n', R2_gru, MAE_gru, RMSE_gru);

if R2_gru > R2_cnn
    fprintf('=> Best: GRU\n');
    best_model = 'GRU'; best_yp = yp_gru; best_net = net_gru;
else
    fprintf('=> Best: CNN\n');
    best_model = 'CNN'; best_yp = yp_cnn; best_net = net_cnn;
end

figure;
subplot(1,2,1);
scatter(ytrue, yp_cnn, 30, 'filled'); grid on; hold on;
plot([min(ytrue) max(ytrue)], [min(ytrue) max(ytrue)], 'k--', 'LineWidth', 1.5);
xlabel('True PWV_{cf} (m/s)'); ylabel('Predicted PWV_{cf} (m/s)');
title(sprintf('CNN: R^2 = %.3f', R2_cnn));

subplot(1,2,2);
scatter(ytrue, yp_gru, 30, 'filled'); grid on; hold on;
plot([min(ytrue) max(ytrue)], [min(ytrue) max(ytrue)], 'k--', 'LineWidth', 1.5);
xlabel('True PWV_{cf} (m/s)'); ylabel('Predicted PWV_{cf} (m/s)');
title(sprintf('GRU: R^2 = %.3f', R2_gru));
save_figure('deep_learning_comparison', 9);

% Find project root and save models
current_dir = pwd;
while ~exist(fullfile(current_dir, 'literature-review.md'), 'file')
    parent_dir = fileparts(current_dir);
    if strcmp(current_dir, parent_dir)
        error('Could not find project root');
    end
    current_dir = parent_dir;
end
models_dir = fullfile(current_dir, 'models');
if ~exist(models_dir, 'dir')
    mkdir(models_dir);
end
% Save performance metrics
metrics.CNN.R2 = R2_cnn;
metrics.CNN.MAE = MAE_cnn;
metrics.CNN.RMSE = RMSE_cnn;
metrics.CNN.training_time = cnn_time;
metrics.GRU.R2 = R2_gru;
metrics.GRU.MAE = MAE_gru;
metrics.GRU.RMSE = RMSE_gru;
metrics.GRU.training_time = gru_time;

save(fullfile(models_dir, 'part9_cnn_gru_models.mat'), 'net_cnn', 'net_gru', 'best_model', 'best_net', 'metrics');

% Model Interpretability Analysis
fprintf('\n--- Running Interpretability Analysis ---\n');
model_interpretability(net_cnn, seqData(testIdx), ytrue, 'CNN');
model_interpretability(net_gru, seqData(testIdx), ytrue, 'GRU');

fprintf('Part 9: Deep learning completed\n');
end