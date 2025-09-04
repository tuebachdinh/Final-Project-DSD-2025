function part12_cross_validation(waves, PWV_cf, waves_augmented, PWV_cf_augmented, k, which_channels)
%PART12_CROSS_VALIDATION K-fold CV for TCN with mixed clean+aug training
if nargin < 5 || isempty(k), k = 5; end
if nargin < 6 || isempty(which_channels), which_channels = 'both'; end
rng(8);

% ------------------ Select & clean rows ------------------
X_ppg_clean  = waves.PPG_Radial;   % [N x T]
X_area_clean = waves.A_Radial;     % [N x T]
y_clean      = PWV_cf(:);

X_ppg_aug  = waves_augmented.PPG_Radial;
X_area_aug = waves_augmented.A_Radial;
y_aug      = PWV_cf_augmented(:);

good_clean = all(isfinite(X_ppg_clean),2) & all(isfinite(X_area_clean),2) & isfinite(y_clean);
good_aug   = all(isfinite(X_ppg_aug),2)   & all(isfinite(X_area_aug),2)   & isfinite(y_aug);
good = good_clean & good_aug;

X_ppg_clean  = X_ppg_clean(good,:);   X_area_clean = X_area_clean(good,:);   y_clean = y_clean(good);
X_ppg_aug    = X_ppg_aug(good,:);     X_area_aug   = X_area_aug(good,:);     y_aug   = y_aug(good);

[N, T] = size(X_ppg_clean);
fprintf('\n=== Part 12: K-fold CV (k=%d) | channels=%s | N=%d, T=%d ===\n', k, which_channels, N, T);

% ------------------ Pack sequences ------------------
seq_clean = pack_seq(X_ppg_clean, X_area_clean, which_channels);
seq_aug   = pack_seq(X_ppg_aug,   X_area_aug,   which_channels);

% ------------------ K-fold indices (subject-wise) ------------------
idx = randperm(N);
fold_sizes = repmat(floor(N/k), 1, k);
rem = N - sum(fold_sizes);  fold_sizes(1:rem) = fold_sizes(1:rem) + 1;

folds = cell(k,1);  startPos = 1;
for f=1:k
    folds{f} = idx(startPos:startPos+fold_sizes(f)-1);
    startPos = startPos + fold_sizes(f);
end

% ------------------ Training options (same as Part 9) ------------------
opts_template = trainingOptions('adam', ...
    'InitialLearnRate', 1e-3, ...
    'MaxEpochs', 70, ...
    'MiniBatchSize', 32, ...
    'Shuffle', 'every-epoch', ...
    'Verbose', true, ...
    'Plots', 'training-progress', ...
    'ValidationFrequency', 20);

% ------------------ Metrics containers ------------------
metr = struct();  metr.fold = (1:k)';
fields = {'R2_clean','MAE_clean','RMSE_clean','R2_aug','MAE_aug','RMSE_aug','train_time_s'};
for fi = 1:numel(fields), metr.(fields{fi}) = nan(k,1); end

% ------------------ Project root + save dir ------------------
current_dir = pwd;
while ~exist(fullfile(current_dir, 'literature-review.md'), 'file')
    parent_dir = fileparts(current_dir);
    if strcmp(current_dir, parent_dir), error('Could not find project root'); end
    current_dir = parent_dir;
end
models_dir = fullfile(current_dir, 'models');
if ~exist(models_dir, 'dir'), mkdir(models_dir); end

% ------------------ Fold loop ------------------
for f = 1:k
    testIdx = folds{f};
    trainIdx_all = setdiff(idx, testIdx);

    % Validation split (clean-only) from train (20%)
    nTrain = numel(trainIdx_all);
    nVal   = max(1, round(0.2 * nTrain));
    p = randperm(nTrain);
    valIdx_local = trainIdx_all(p(1:nVal));
    trIdx_local  = trainIdx_all(p(nVal+1:end));

    % Training set: concat clean + aug, then shuffle
    seqTrain = [seq_clean(trIdx_local); seq_aug(trIdx_local)];
    yTrain   = [y_clean(trIdx_local);    y_aug(trIdx_local)];
    sh = randperm(numel(seqTrain));
    seqTrain = seqTrain(sh);  yTrain = yTrain(sh);

    % Validation (clean-only)
    seqVal = seq_clean(valIdx_local);
    yVal   = y_clean(valIdx_local);

    % Tests (both versions of held-out subjects)
    seqTest_clean = seq_clean(testIdx);   yTest_clean = y_clean(testIdx);
    seqTest_aug   = seq_aug(testIdx);     yTest_aug   = y_aug(testIdx);

    % Build TCN (same architecture)
    nCh = size(seq_clean{1},1);
    F   = 55;
    lgraph_tcn = create_tcn_lgraph(nCh, F);

    % Train (same optimization)
    opts = opts_template;  opts.ValidationData = {seqVal, yVal};
    fprintf('\n[Fold %d/%d] Training TCN ...\n', f, k);
    tic;  net_tcn = trainNetwork(seqTrain, yTrain, lgraph_tcn, opts);  train_time = toc;
    
    save(fullfile(models_dir, sprintf('tcn_fold%d.mat', f)), 'net_tcn');

    % Evaluate
    [R2_c, MAE_c, RMSE_c] = eval_metrics(net_tcn, seqTest_clean, yTest_clean);
    [R2_a, MAE_a, RMSE_a] = eval_metrics(net_tcn, seqTest_aug,   yTest_aug);

    % Store
    metr.R2_clean(f)     = R2_c;   metr.MAE_clean(f)  = MAE_c;   metr.RMSE_clean(f)  = RMSE_c;
    metr.R2_aug(f)       = R2_a;   metr.MAE_aug(f)    = MAE_a;   metr.RMSE_aug(f)    = RMSE_a;
    metr.train_time_s(f) = train_time;

    fprintf('[Fold %d] Clean: R^2=%.3f | MAE=%.3f | RMSE=%.3f | time=%.1fs\n', f, R2_c, MAE_c, RMSE_c, train_time);
    fprintf('[Fold %d]  Aug : R^2=%.3f | MAE=%.3f | RMSE=%.3f\n', f, R2_a, MAE_a, RMSE_a);
end

% ------------------ Aggregate & print ------------------
mean_std = @(v) deal(mean(v,'omitnan'), std(v,'omitnan'));

[mR2c, sR2c]    = mean_std(metr.R2_clean);
[mMAEc, sMAEc]  = mean_std(metr.MAE_clean);
[mRMSEc, sRMSEc]= mean_std(metr.RMSE_clean);

[mR2a, sR2a]    = mean_std(metr.R2_aug);
[mMAEa, sMAEa]  = mean_std(metr.MAE_aug);
[mRMSEa, sRMSEa]= mean_std(metr.RMSE_aug);

[mTIME, sTIME]  = mean_std(metr.train_time_s);

fprintf('\n=== K-fold CV Summary (k=%d | %s) ===\n', k, which_channels);
fprintf('Clean  : R^2 = %.3f ± %.3f | MAE = %.3f ± %.3f | RMSE = %.3f ± %.3f\n', mR2c, sR2c, mMAEc, sMAEc, mRMSEc, sRMSEc);
fprintf('Augment: R^2 = %.3f ± %.3f | MAE = %.3f ± %.3f | RMSE = %.3f ± %.3f\n', mR2a, sR2a, mMAEa, sMAEa, mRMSEa, sRMSEa);
fprintf('Train time per fold: %.1fs ± %.1fs\n', mTIME, sTIME);

% ------------------ Save ------------------
save_name = fullfile(models_dir, sprintf('part12_tcn_kfold_%dfold_%s.mat', k, lower(which_channels)));
results_summary = struct( ...
    'k', k, ...
    'which_channels', which_channels, ...
    'per_fold', metr, ...
    'summary', struct( ...
        'clean', struct('R2_mean',mR2c,'R2_std',sR2c,'MAE_mean',mMAEc,'MAE_std',sMAEc,'RMSE_mean',mRMSEc,'RMSE_std',sRMSEc), ...
        'aug',   struct('R2_mean',mR2a,'R2_std',sR2a,'MAE_mean',mMAEa,'MAE_std',sMAEa,'RMSE_mean',mRMSEa,'RMSE_std',sRMSEa), ...
        'train_time_s', struct('mean',mTIME,'std',sTIME) ...
    ) ...
);
save(save_name, 'results_summary');
fprintf('Saved: %s\n', save_name);
fprintf('Part 12 complete.\n');

end % ===== end main function =====


% =============== LOCAL HELPER FUNCTIONS (keep below) ===============

function S = pack_seq(Xp, Xa, mode)
Xp_z = znorm(Xp);  Xa_z = znorm(Xa);
S = cell(size(Xp,1),1);
for ii=1:size(Xp,1)
    switch lower(mode)
        case 'both', S{ii} = [Xp_z(ii,:); Xa_z(ii,:)];
        case 'ppg',  S{ii} = Xp_z(ii,:);
        case 'area', S{ii} = Xa_z(ii,:);
        otherwise, error('which_channels must be both|ppg|area');
    end
end
end

function Xz = znorm(X)
Xz = (X - mean(X,2)) ./ max(std(X,[],2), eps);
end

function [R2, MAE, RMSE] = eval_metrics(net, Xcell, ytrue)
ypred = predict(net, Xcell);
resid = ytrue - ypred;
SSres = sum(resid.^2);
SStot = sum((ytrue - mean(ytrue)).^2);
R2    = 1 - SSres / SStot;
MAE   = mean(abs(resid));
RMSE  = sqrt(mean(resid.^2));
end

function lgraph = create_tcn_lgraph(nCh, F)
lgraph = layerGraph();

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

% Add & connect
for L = [inp, b1_conv1, b1_relu1, b1_drop1, b1_conv2, b1_relu2, b1_skip, b1_add, b1_out, ...
          b2_conv1, b2_relu1, b2_drop1, b2_conv2, b2_relu2, b2_skip, b2_add, b2_out, ...
          gap, fc1, reluH, fc2, reg]
    lgraph = addLayers(lgraph, L);
end

lgraph = connectLayers(lgraph,'input','b1_conv1');
lgraph = connectLayers(lgraph,'b1_conv1','b1_relu1');
lgraph = connectLayers(lgraph,'b1_relu1','b1_drop1');
lgraph = connectLayers(lgraph,'b1_drop1','b1_conv2');
lgraph = connectLayers(lgraph,'b1_conv2','b1_relu2');
lgraph = connectLayers(lgraph,'input','b1_skip');
lgraph = connectLayers(lgraph,'b1_relu2','b1_add/in1');
lgraph = connectLayers(lgraph,'b1_skip','b1_add/in2');
lgraph = connectLayers(lgraph,'b1_add','b1_out');

lgraph = connectLayers(lgraph,'b1_out','b2_conv1');
lgraph = connectLayers(lgraph,'b1_out','b2_skip');
lgraph = connectLayers(lgraph,'b2_conv1','b2_relu1');
lgraph = connectLayers(lgraph,'b2_relu1','b2_drop1');
lgraph = connectLayers(lgraph,'b2_drop1','b2_conv2');
lgraph = connectLayers(lgraph,'b2_conv2','b2_relu2');
lgraph = connectLayers(lgraph,'b2_relu2','b2_add/in1');
lgraph = connectLayers(lgraph,'b2_skip','b2_add/in2');
lgraph = connectLayers(lgraph,'b2_add','b2_out');

lgraph = connectLayers(lgraph,'b2_out','gap');
lgraph = connectLayers(lgraph,'gap','fc1');
lgraph = connectLayers(lgraph,'fc1','reluH');
lgraph = connectLayers(lgraph,'reluH','fc2');
lgraph = connectLayers(lgraph,'fc2','regression');
end
