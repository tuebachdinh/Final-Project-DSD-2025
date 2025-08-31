function part9_deep_learning(waves, PWV_cf, waves_augmented, PWV_cf_augmented)
%PART9_DEEP_LEARNING Compare CNN vs GRU vs TCN (with ability to reuse CNN/GRU)
addpath('../utils/others');
addpath('../utils/deep_learning');

rng(8);

% ------------------ DATA SELECTION ------------------
% Current setting: augmented data (as in your code)
X_ppg  = waves_augmented.PPG_Radial;   % [N x T]
X_area = waves_augmented.A_Radial;     % [N x T]
y      = PWV_cf_augmented(:);
data_tag = 'augmented';

% If you switch to clean data later, uncomment below and set data_tag='clean'
% X_ppg  = waves.PPG_Radial;
% X_area = waves.A_Radial;
% y      = PWV_cf(:);
% data_tag = 'clean';

fprintf('\n=== Part 9: Deep Learning (%s data) ===\n', data_tag);
fprintf('Total subjects: %d\n', numel(y));

good = all(isfinite(X_ppg),2) & all(isfinite(X_area),2) & isfinite(y);
X_ppg  = X_ppg(good,:); 
X_area = X_area(good,:);
y      = y(good);

N = size(X_ppg,1);
T = size(X_ppg,2);

% ---------- Choose input channels ----------
% options: 'both' (default), 'ppg', or 'area'
which_channels = 'area';  % <-- keep your current setting

switch lower(which_channels)
    case 'both', nCh = 2; chan_tag = 'both';
    case 'ppg',  nCh = 1; chan_tag = 'PPG';
    case 'area', nCh = 1; chan_tag = 'Area';
    otherwise, error('which_channels must be ''both'', ''ppg'', or ''area''');
end

% (recommended) per-sequence z-norm both channels
X_ppg_z  = (X_ppg  - mean(X_ppg,2))  ./ max(std(X_ppg,[],2),  eps);
X_area_z = (X_area - mean(X_area,2)) ./ max(std(X_area,[],2), eps);

% Build seqData cell
seqData = cell(N,1);
for i = 1:N
    switch lower(which_channels)
        case 'both', seqData{i} = [X_ppg_z(i,:); X_area_z(i,:)];
        case 'ppg',  seqData{i} = X_ppg_z(i,:);
        case 'area', seqData{i} = X_area_z(i,:);
    end
end

% ------------------ SPLIT ------------------
idx = randperm(N);
nTrain = round(0.6*N);
nVal   = round(0.2*N);

valIdx   = idx(1:nVal);
trainIdx = idx(nVal+1:nVal+nTrain);
testIdx  = idx(nVal+nTrain+1:end);

% ------------------ TRAINING OPTIONS ------------------
opts = trainingOptions('adam', ...
    'InitialLearnRate', 1e-3, ...
    'MaxEpochs', 70, ...
    'MiniBatchSize', 32, ...
    'Shuffle', 'every-epoch', ...
    'Verbose', true, ...
    'Plots', 'training-progress', ...
    'ValidationData', {seqData(valIdx), y(valIdx)}, ...
    'ValidationFrequency', 20);

% ------------------ PROJECT ROOT / MODELS DIR ------------------
current_dir = pwd;
while ~exist(fullfile(current_dir, 'literature-review.md'), 'file')
    parent_dir = fileparts(current_dir);
    if strcmp(current_dir, parent_dir), error('Could not find project root'); end
    current_dir = parent_dir;
end
models_dir = fullfile(current_dir, 'models');
if ~exist(models_dir, 'dir'), mkdir(models_dir); end

% ------------------ CONTROL: reuse CNN/GRU ------------------
TRAIN_CNN = false;   % <- do NOT retrain CNN
TRAIN_GRU = false;   % <- do NOT retrain GRU
TRAIN_TCN = true;    % <- train TCN

pretrained_file = fullfile(models_dir, sprintf('part9_models_%s_%s.mat', data_tag, chan_tag));
if exist(pretrained_file,'file')
    S_pre = load(pretrained_file);
else
    S_pre = struct();
end

% =======================================================
%                         CNN
% =======================================================
cnn_time = NaN; R2_cnn = NaN; MAE_cnn = NaN; RMSE_cnn = NaN;
if ~TRAIN_CNN && isfield(S_pre,'net_cnn')
    net_cnn = S_pre.net_cnn;
    fprintf('\n--- CNN: using preloaded model from %s ---\n', pretrained_file);
else
    fprintf('\n--- Training CNN Model ---\n');
    layers_cnn = [
        sequenceInputLayer(nCh, 'MinLength', T, 'Name', 'input')
        convolution1dLayer(10, 48, 'Padding', 'same')
        reluLayer
        convolution1dLayer(10, 96, 'Padding', 'same')
        reluLayer
        globalAveragePooling1dLayer
        fullyConnectedLayer(48)
        reluLayer
        fullyConnectedLayer(1)
        regressionLayer];
    tic;
    net_cnn = trainNetwork(seqData(trainIdx), y(trainIdx), layers_cnn, opts);
    cnn_time = toc;
end

% Evaluate CNN if available
if exist('net_cnn','var')
    ytrue = y(testIdx);
    yp_cnn   = predict(net_cnn, seqData(testIdx));
    resid_cnn = ytrue - yp_cnn;
    R2_cnn    = 1 - sum(resid_cnn.^2) / sum((ytrue - mean(ytrue)).^2);
    MAE_cnn   = mean(abs(resid_cnn));
    RMSE_cnn  = sqrt(mean(resid_cnn.^2));
end

% =======================================================
%                         GRU
% =======================================================
gru_time = NaN; R2_gru = NaN; MAE_gru = NaN; RMSE_gru = NaN;
if ~TRAIN_GRU && isfield(S_pre,'net_gru')
    net_gru = S_pre.net_gru;
    fprintf('\n--- GRU: using preloaded model from %s ---\n', pretrained_file);
else
    fprintf('\n--- Training GRU Model ---\n');
    layers_gru = [
        sequenceInputLayer(nCh, 'Name', 'input')
        gruLayer(96,'OutputMode','sequence')
        dropoutLayer(0.15)
        gruLayer(48,'OutputMode','last')
        fullyConnectedLayer(48)
        reluLayer
        fullyConnectedLayer(1)
        regressionLayer];
    tic;
    net_gru = trainNetwork(seqData(trainIdx), y(trainIdx), layers_gru, opts);
    gru_time = toc;
end

% Evaluate GRU if available
if exist('net_gru','var')
    ytrue = y(testIdx);
    yp_gru   = predict(net_gru, seqData(testIdx));
    resid_gru = ytrue - yp_gru;
    R2_gru    = 1 - sum(resid_gru.^2) / sum((ytrue - mean(ytrue)).^2);
    MAE_gru   = mean(abs(resid_gru));
    RMSE_gru  = sqrt(mean(resid_gru.^2));
end

% =======================================================
%                         TCN
% 2 residual blocks with dilations (1,2) and (4,8), F=64 filters.
% ~65k parameters total (close to CNN/GRU budget).
% =======================================================
fprintf('\n--- Training TCN Model ---\n');
lgraph_tcn = create_tcn_lgraph(nCh, 55);  % F=64
tic;
net_tcn = trainNetwork(seqData(trainIdx), y(trainIdx), lgraph_tcn, opts);
tcn_time = toc;

% Evaluate TCN
ytrue  = y(testIdx);
yp_tcn = predict(net_tcn, seqData(testIdx));
resid_tcn = ytrue - yp_tcn;
R2_tcn   = 1 - sum(resid_tcn.^2) / sum((ytrue - mean(ytrue)).^2);
MAE_tcn  = mean(abs(resid_tcn));
RMSE_tcn = sqrt(mean(resid_tcn.^2));

% ------------------ PRINT RESULTS ------------------
fprintf('\n=== Part 9 Results (%s | %s) ===\n', data_tag, chan_tag);
if exist('net_cnn','var')
    fprintf('CNN:  R^2 = %.3f | MAE = %.3f | RMSE = %.3f | time = %.1fs\n', R2_cnn, MAE_cnn, RMSE_cnn, cnn_time);
end
if exist('net_gru','var')
    fprintf('GRU:  R^2 = %.3f | MAE = %.3f | RMSE = %.3f | time = %.1fs\n', R2_gru, MAE_gru, RMSE_gru, gru_time);
end
fprintf('TCN:  R^2 = %.3f | MAE = %.3f | RMSE = %.3f | time = %.1fs\n', R2_tcn, MAE_tcn, RMSE_tcn, tcn_time);

% Decide best among the models available
R2s = [-Inf, -Inf, R2_tcn]; names = {'CNN','GRU','TCN'}; nets = {[],[],net_tcn}; yps = {[],[],yp_tcn};
if exist('net_cnn','var'), R2s(1)=R2_cnn; nets{1}=net_cnn; yps{1}=predict(net_cnn,seqData(testIdx)); end
if exist('net_gru','var'), R2s(2)=R2_gru; nets{2}=net_gru; yps{2}=predict(net_gru,seqData(testIdx)); end
[~,best_idx] = max(R2s);
best_model = names{best_idx};
best_net   = nets{best_idx};
best_yp    = yps{best_idx};

fprintf('=> Best: %s\n', best_model);

% ------------------ PLOTS ------------------
% Plot available models side-by-side
avail = find(~isinf(R2s));
nPlot = numel(avail);
figure;
for k = 1:nPlot
    m = avail(k);
    subplot(1,nPlot,k);
    switch m
        case 1, yp = predict(net_cnn, seqData(testIdx)); ttl = sprintf('CNN: R^2=%.3f', R2_cnn);
        case 2, yp = predict(net_gru, seqData(testIdx)); ttl = sprintf('GRU: R^2=%.3f', R2_gru);
        case 3, yp = yp_tcn;                              ttl = sprintf('TCN: R^2=%.3f', R2_tcn);
    end
    scatter(ytrue, yp, 30, 'filled'); grid on; hold on;
    plot([min(ytrue) max(ytrue)], [min(ytrue) max(ytrue)], 'k--', 'LineWidth', 1.2);
    xlabel('True PWV_{cf} (m/s)'); ylabel('Predicted PWV_{cf} (m/s)');
    title(ttl);
end
save_figure(sprintf('comparison_%s_%s', data_tag, lower(chan_tag)), 9);

% ------------------ SAVE ------------------
metrics = struct();
if exist('net_cnn','var')
    metrics.CNN.R2 = R2_cnn; metrics.CNN.MAE = MAE_cnn; metrics.CNN.RMSE = RMSE_cnn; metrics.CNN.training_time = cnn_time;
end
if exist('net_gru','var')
    metrics.GRU.R2 = R2_gru; metrics.GRU.MAE = MAE_gru; metrics.GRU.RMSE = RMSE_gru; metrics.GRU.training_time = gru_time;
end
metrics.TCN.R2 = R2_tcn; metrics.TCN.MAE = MAE_tcn; metrics.TCN.RMSE = RMSE_tcn; metrics.TCN.training_time = tcn_time;

test_data.seqData = seqData(testIdx);
test_data.ytrue   = ytrue;

save_name = fullfile(models_dir, sprintf('part9_models_%s_%s_with_TCN.mat', data_tag, chan_tag));
if exist('net_cnn','var') && exist('net_gru','var')
    save(save_name, 'net_cnn', 'net_gru', 'net_tcn', 'best_model', 'best_net', 'metrics', 'test_data');
elseif exist('net_cnn','var')
    save(save_name, 'net_cnn', 'net_tcn', 'best_model', 'best_net', 'metrics', 'test_data');
elseif exist('net_gru','var')
    save(save_name, 'net_gru', 'net_tcn', 'best_model', 'best_net', 'metrics', 'test_data');
else
    save(save_name, 'net_tcn', 'best_model', 'best_net', 'metrics', 'test_data');
end

fprintf('Saved: %s\n', save_name);
fprintf('Part 9 with TCN complete.\n');

end

% =======================================================
%                  TCN LAYER GRAPH (2 blocks)
% Each block: Conv(k=5,d=•) -> ReLU -> Dropout -> Conv(k=5,d=•) -> ReLU
% + 1x1 Conv skip; Residual add; Then GAP -> FC(48) -> ReLU -> FC(1)
% Filters F=64 → ~65k params (close to CNN/GRU budget).
% =======================================================
function lgraph = create_tcn_lgraph(nCh, F)
% TCN with two residual blocks: dilations (1,2) and (4,8)
% Head: GAP -> FC(48) -> ReLU -> FC(1) -> regression
% Use F=59 to stay ~60k params (close to CNN/GRU budget).

lgraph = layerGraph();

% ---- define layers ----
inp      = sequenceInputLayer(nCh, 'Name','input');

% Block 1
b1_conv1 = convolution1dLayer(5, F,'Padding','same','DilationFactor',1,'Name','b1_conv1');
b1_relu1 = reluLayer('Name','b1_relu1');
b1_drop1 = dropoutLayer(0.1,'Name','b1_drop1');
b1_conv2 = convolution1dLayer(5, F,'Padding','same','DilationFactor',2,'Name','b1_conv2');
b1_relu2 = reluLayer('Name','b1_relu2');
b1_skip  = convolution1dLayer(1, F,'Padding','same','Name','b1_skip');
b1_add   = additionLayer(2,'Name','b1_add');
b1_out   = reluLayer('Name','b1_out');

% Block 2
b2_conv1 = convolution1dLayer(5, F,'Padding','same','DilationFactor',4,'Name','b2_conv1');
b2_relu1 = reluLayer('Name','b2_relu1');
b2_drop1 = dropoutLayer(0.1,'Name','b2_drop1');
b2_conv2 = convolution1dLayer(5, F,'Padding','same','DilationFactor',8,'Name','b2_conv2');
b2_relu2 = reluLayer('Name','b2_relu2');
b2_skip  = convolution1dLayer(1, F,'Padding','same','Name','b2_skip');
b2_add   = additionLayer(2,'Name','b2_add');
b2_out   = reluLayer('Name','b2_out');

% Head
gap   = globalAveragePooling1dLayer('Name','gap');
fc1   = fullyConnectedLayer(48,'Name','fc1');
reluH = reluLayer('Name','reluH');
fc2   = fullyConnectedLayer(1,'Name','fc2');
reg   = regressionLayer('Name','regression');

% ---- add layers individually (avoid accidental auto-chaining) ----
for L = [inp, b1_conv1, b1_relu1, b1_drop1, b1_conv2, b1_relu2, b1_skip, b1_add, b1_out, ...
          b2_conv1, b2_relu1, b2_drop1, b2_conv2, b2_relu2, b2_skip, b2_add, b2_out, ...
          gap, fc1, reluH, fc2, reg]
    lgraph = addLayers(lgraph, L);
end

% ---- safe connections (only connect if not already connected) ----
lgraph = safeConnect(lgraph,'input','b1_conv1');
lgraph = safeConnect(lgraph,'b1_conv1','b1_relu1');
lgraph = safeConnect(lgraph,'b1_relu1','b1_drop1');
lgraph = safeConnect(lgraph,'b1_drop1','b1_conv2');
lgraph = safeConnect(lgraph,'b1_conv2','b1_relu2');

lgraph = safeConnect(lgraph,'input','b1_skip');
lgraph = safeConnect(lgraph,'b1_relu2','b1_add/in1');
lgraph = safeConnect(lgraph,'b1_skip','b1_add/in2');
lgraph = safeConnect(lgraph,'b1_add','b1_out');

lgraph = safeConnect(lgraph,'b1_out','b2_conv1');
lgraph = safeConnect(lgraph,'b1_out','b2_skip');
lgraph = safeConnect(lgraph,'b2_conv1','b2_relu1');
lgraph = safeConnect(lgraph,'b2_relu1','b2_drop1');
lgraph = safeConnect(lgraph,'b2_drop1','b2_conv2');
lgraph = safeConnect(lgraph,'b2_conv2','b2_relu2');

lgraph = safeConnect(lgraph,'b2_relu2','b2_add/in1');
lgraph = safeConnect(lgraph,'b2_skip','b2_add/in2');
lgraph = safeConnect(lgraph,'b2_add','b2_out');

lgraph = safeConnect(lgraph,'b2_out','gap');
lgraph = safeConnect(lgraph,'gap','fc1');
lgraph = safeConnect(lgraph,'fc1','reluH');
lgraph = safeConnect(lgraph,'reluH','fc2');
lgraph = safeConnect(lgraph,'fc2','regression');

end

% ---------- helper ----------
function lgraph = safeConnect(lgraph, src, dst)
C = lgraph.Connections;
if isempty(C)
    need = true;
else
    need = ~any(strcmp(C.Source, src) & strcmp(C.Destination, dst));
end
if need
    lgraph = connectLayers(lgraph, src, dst);
end
end
