function part9_deep_learning(waves, PWV_cf, waves_augmented, PWV_cf_augmented, train_data_tag, which_channels)
%PART9_DEEP_LEARNING Compare CNN vs GRU vs TCN
% Train on 'clean' or 'augmented' (train_data_tag), but ALWAYS validate & test on CLEAN.
% which_channels: 'both' | 'ppg' | 'area'

addpath('../utils/others');
addpath('../utils/deep_learning');
rng(8);

% ---------- defaults ----------
if nargin < 5 || isempty(train_data_tag), train_data_tag = 'clean'; end
if nargin < 6 || isempty(which_channels), which_channels = 'both'; end
assert(ismember(lower(train_data_tag), {'clean','augmented'}), 'train_data_tag must be clean|augmented');
assert(ismember(lower(which_channels), {'both','ppg','area'}), 'which_channels must be both|ppg|area');

fprintf('\n=== Part 9: Deep Learning (train=%s | val/test=clean) ===\n', lower(train_data_tag));

% ---------- channel setup ----------
switch lower(which_channels)
    case 'both', nCh = 2; chan_tag = 'both';
    case 'ppg',  nCh = 1; chan_tag = 'PPG';
    case 'area', nCh = 1; chan_tag = 'Area';
end

% ---------- CLEAN dataset (always used for validation & test) ----------
Xc_ppg  = waves.PPG_Radial;
Xc_area = waves.A_Radial;
yc      = PWV_cf(:);

good_c  = all(isfinite(Xc_ppg),2) & all(isfinite(Xc_area),2) & isfinite(yc);
Xc_ppg  = Xc_ppg(good_c,:); 
Xc_area = Xc_area(good_c,:);
yc      = yc(good_c);

% per-sequence z-norm
Xc_ppg_z  = (Xc_ppg  - mean(Xc_ppg,2))  ./ max(std(Xc_ppg,[],2),  eps);
Xc_area_z = (Xc_area - mean(Xc_area,2)) ./ max(std(Xc_area,[],2), eps);

Nc = size(Xc_ppg_z,1);
Tc = size(Xc_ppg_z,2);

seqClean = cell(Nc,1);
for i = 1:Nc
    switch lower(which_channels)
        case 'both', seqClean{i} = [Xc_ppg_z(i,:); Xc_area_z(i,:)];
        case 'ppg',  seqClean{i} = Xc_ppg_z(i,:);
        case 'area', seqClean{i} = Xc_area_z(i,:);
    end
end

% ---------- AUGMENTED dataset (used for training iff train_data_tag='augmented') ----------
Xa_ppg  = waves_augmented.PPG_Radial;
Xa_area = waves_augmented.A_Radial;
ya      = PWV_cf_augmented(:);

if ~isempty(Xa_ppg) && ~isempty(Xa_area) && ~isempty(ya)
    good_a  = all(isfinite(Xa_ppg),2) & all(isfinite(Xa_area),2) & isfinite(ya);
    Xa_ppg  = Xa_ppg(good_a,:); 
    Xa_area = Xa_area(good_a,:);
    ya      = ya(good_a);

    Xa_ppg_z  = (Xa_ppg  - mean(Xa_ppg,2))  ./ max(std(Xa_ppg,[],2),  eps);
    Xa_area_z = (Xa_area - mean(Xa_area,2)) ./ max(std(Xa_area,[],2), eps);

    Na = size(Xa_ppg_z,1);
    seqAug = cell(Na,1);
    for i = 1:Na
        switch lower(which_channels)
            case 'both', seqAug{i} = [Xa_ppg_z(i,:); Xa_area_z(i,:)];
            case 'ppg',  seqAug{i} = Xa_ppg_z(i,:);
            case 'area', seqAug{i} = Xa_area_z(i,:);
        end
    end
else
    Na = 0; seqAug = {}; ya = [];
end

% ------------------ SPLITS (on CLEAN only) ------------------
idxc    = randperm(Nc);
nTrainC = round(0.6*Nc);
nValC   = round(0.2*Nc);
valIdx  = idxc(1:nValC);
trainIdxC = idxc(nValC+1:nValC+nTrainC);
testIdx = idxc(nValC+nTrainC+1:end);

% ------------------ TRAIN SET CHOICE ------------------
if strcmpi(train_data_tag,'augmented')
    if Na==0, error('Augmented data requested for training, but none provided.'); end
    seqTrain = seqAug;
    yTrain   = ya;
else
    seqTrain = seqClean(trainIdxC);
    yTrain   = yc(trainIdxC);
end
seqVal = seqClean(valIdx);   yVal = yc(valIdx);     % always clean
ytrue  = yc(testIdx);        % always clean test
T = Tc; % common length

% ------------------ TRAINING OPTIONS ------------------
opts = trainingOptions('adam', ...
    'InitialLearnRate', 1e-3, ...
    'MaxEpochs', 70, ...
    'MiniBatchSize', 32, ...
    'Shuffle', 'every-epoch', ...
    'Verbose', true, ...
    'Plots', 'training-progress', ...
    'ValidationData', {seqVal, yVal}, ...
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
TRAIN_CNN = true;
TRAIN_GRU = true;
TRAIN_TCN = true;

tag_train = lower(train_data_tag);
pretrained_file = fullfile(models_dir, sprintf('part9_models_%s_%s.mat', tag_train, chan_tag));
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
    net_cnn = trainNetwork(seqTrain, yTrain, layers_cnn, opts);
    cnn_time = toc;
end

if exist('net_cnn','var')
    yp_cnn   = predict(net_cnn, seqClean(testIdx));
    resid_cnn = ytrue - yp_cnn;
    R2_cnn    = 1 - sum(resid_cnn.^2) / sum((ytrue - mean(ytrue)).^2);
    MAE_cnn   = mean(abs(resid_cnn));
    RMSE_cnn  = sqrt(mean(resid_cnn.^2));
    if exist('S_pre','var') && isfield(S_pre,'metrics') && isfield(S_pre.metrics,'CNN') ...
            && isfield(S_pre.metrics.CNN,'training_time') && ~TRAIN_CNN
        cnn_time = S_pre.metrics.CNN.training_time;
    end
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
    net_gru = trainNetwork(seqTrain, yTrain, layers_gru, opts);
    gru_time = toc;
end

if exist('net_gru','var')
    yp_gru   = predict(net_gru, seqClean(testIdx));
    resid_gru = ytrue - yp_gru;
    R2_gru    = 1 - sum(resid_gru.^2) / sum((ytrue - mean(ytrue)).^2);
    MAE_gru   = mean(abs(resid_gru));
    RMSE_gru  = sqrt(mean(resid_gru.^2));
    if exist('S_pre','var') && isfield(S_pre,'metrics') && isfield(S_pre.metrics,'GRU') ...
            && isfield(S_pre.metrics.GRU,'training_time') && ~TRAIN_GRU
        gru_time = S_pre.metrics.GRU.training_time;
    end
end

% =======================================================
%                         TCN
% =======================================================
fprintf('\n--- Training TCN Model ---\n');
lgraph_tcn = create_tcn_lgraph(nCh, 55);
tic;
net_tcn = trainNetwork(seqTrain, yTrain, lgraph_tcn, opts);
tcn_time = toc;

% Evaluate TCN (on CLEAN test)
yp_tcn = predict(net_tcn, seqClean(testIdx));
resid_tcn = ytrue - yp_tcn;
R2_tcn   = 1 - sum(resid_tcn.^2) / sum((ytrue - mean(ytrue)).^2);
MAE_tcn  = mean(abs(resid_tcn));
RMSE_tcn = sqrt(mean(resid_tcn.^2));

% ------------------ PRINT RESULTS ------------------
fprintf('\n=== Part 9 Results (train=%s | val/test=clean | %s) ===\n', tag_train, chan_tag);
if exist('net_cnn','var')
    fprintf('CNN:  R^2 = %.3f | MAE = %.3f | RMSE = %.3f | time = %.1fs\n', R2_cnn, MAE_cnn, RMSE_cnn, cnn_time);
end
if exist('net_gru','var')
    fprintf('GRU:  R^2 = %.3f | MAE = %.3f | RMSE = %.3f | time = %.1fs\n', R2_gru, MAE_gru, RMSE_gru, gru_time);
end
fprintf('TCN:  R^2 = %.3f | MAE = %.3f | RMSE = %.3f | time = %.1fs\n', R2_tcn, MAE_tcn, RMSE_tcn, tcn_time);

% Decide best
R2s = [-Inf, -Inf, R2_tcn]; names = {'CNN','GRU','TCN'}; nets = {[],[],net_tcn};
if exist('net_cnn','var'), R2s(1)=R2_cnn; nets{1}=net_cnn; end
if exist('net_gru','var'), R2s(2)=R2_gru; nets{2}=net_gru; end
[~,best_idx] = max(R2s);
best_model = names{best_idx};
best_net   = nets{best_idx};
fprintf('=> Best: %s\n', best_model);

% ------------------ PLOTS (CLEAN test only) ------------------
nPlot = sum(~isinf(R2s));
figure;
kp = 0;
if exist('net_cnn','var')
    kp=kp+1; subplot(1,nPlot,kp);
    yp = predict(net_cnn, seqClean(testIdx)); ttl = sprintf('CNN: R^2=%.3f', R2_cnn);
    scatter(ytrue, yp, 30, 'filled'); grid on; hold on;
    plot([min(ytrue) max(ytrue)], [min(ytrue) max(ytrue)], 'k--', 'LineWidth', 1.2);
    xlabel('True PWV_{cf} (m/s)'); ylabel('Predicted PWV_{cf} (m/s)'); title(ttl);
end
if exist('net_gru','var')
    kp=kp+1; subplot(1,nPlot,kp);
    yp = predict(net_gru, seqClean(testIdx)); ttl = sprintf('GRU: R^2=%.3f', R2_gru);
    scatter(ytrue, yp, 30, 'filled'); grid on; hold on;
    plot([min(ytrue) max(ytrue)], [min(ytrue) max(ytrue)], 'k--', 'LineWidth', 1.2);
    xlabel('True PWV_{cf} (m/s)'); ylabel('Predicted PWV_{cf} (m/s)'); title(ttl);
end
kp=kp+1; subplot(1,nPlot,kp);
yp = yp_tcn; ttl = sprintf('TCN: R^2=%.3f', R2_tcn);
scatter(ytrue, yp, 30, 'filled'); grid on; hold on;
plot([min(ytrue) max(ytrue)], [min(ytrue) max(ytrue)], 'k--', 'LineWidth', 1.2);
xlabel('True PWV_{cf} (m/s)'); ylabel('Predicted PWV_{cf} (m/s)'); title(ttl);
save_figure(sprintf('comparison_%s_%s', tag_train, lower(chan_tag)), 9);

% ------------------ SAVE ------------------
metrics = struct();
if exist('net_cnn','var')
    metrics.CNN = struct('R2',R2_cnn,'MAE',MAE_cnn,'RMSE',RMSE_cnn,'training_time',cnn_time);
end
if exist('net_gru','var')
    metrics.GRU = struct('R2',R2_gru,'MAE',MAE_gru,'RMSE',RMSE_gru,'training_time',gru_time);
end
metrics.TCN = struct('R2',R2_tcn,'MAE',MAE_tcn,'RMSE',RMSE_tcn,'training_time',tcn_time);

test_data.seqData = seqClean(testIdx);
test_data.ytrue   = ytrue;

save_name = fullfile(models_dir, sprintf('part9_models_%s_%s.mat', tag_train, chan_tag));
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
fprintf('Part 9 complete.\n');

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

% Receptive field = 1 + (5-1)(1+2+4+8) = 61 time steps 
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
