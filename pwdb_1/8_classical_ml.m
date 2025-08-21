function part8_classical_ml(waves_augmented, PWV_cf_augmented, fs)
%PART8_CLASSICAL_ML Classical ML with augmented data

fprintf('\n=== Extracting comprehensive features from augmented data ===\n');

Nsubj_aug = size(waves_augmented.PPG_Radial, 1);
PPG_R_aug = waves_augmented.PPG_Radial;
A_R_aug = waves_augmented.A_Radial;

% Extract PPG classical features
RI_aug = nan(Nsubj_aug,1); SI_aug = nan(Nsubj_aug,1); AGImod_aug = nan(Nsubj_aug,1);
for ii = 1:Nsubj_aug
    try
        S_temp.v = PPG_R_aug(ii,:)';
        S_temp.fs = fs;
        options_temp.do_plot = false; options_temp.verbose = false;
        [pw_inds_temp, ~, ~, ~] = PulseAnalyse(S_temp, options_temp);
        if ~isempty(pw_inds_temp)
            if isfield(pw_inds_temp, 'RI'), RI_aug(ii) = pw_inds_temp.RI.v; end
            if isfield(pw_inds_temp, 'SI'), SI_aug(ii) = pw_inds_temp.SI.v; end
            if isfield(pw_inds_temp, 'AGI_mod'), AGImod_aug(ii) = pw_inds_temp.AGI_mod.v; end
        end
    catch, continue; end
end

% Area features
Amax_aug = max(A_R_aug, [], 2);
Amin_aug = min(A_R_aug, [], 2);
Amean_aug = mean(A_R_aug, 2);

% Build feature table
y_aug = PWV_cf_augmented;
X = [RI_aug, SI_aug, AGImod_aug, Amax_aug, Amin_aug, Amean_aug];
good = all(isfinite(X), 2) & isfinite(y_aug);
X = X(good, :);
y_aug = y_aug(good);

N = size(X, 1);
cv = cvpartition(N, 'HoldOut', 0.2);
Xtrain = X(training(cv), :);
Ytrain = y_aug(training(cv));
Xtest = X(test(cv), :);
Ytest = y_aug(test(cv));

[Ztrain, muX, stdX] = zscore(Xtrain);
Ztest = (Xtest - muX) ./ stdX;

% Models
results = struct();

% Ridge Regression
ridgeTrained = fitrlinear(Ztrain, Ytrain, ...
    'Learner','leastsquares', 'Regularization','ridge', ...
    'Solver','lbfgs');
yp_ridge = predict(ridgeTrained, Ztest);
results.Ridge = evalReg(Ytest, yp_ridge, 'Ridge');

% Tree
treeMdl = fitrtree(Ztrain, Ytrain, 'MinLeafSize', 100, 'MaxNumSplits', 10);
yp_tree = predict(treeMdl, Ztest);
results.Tree = evalReg(Ytest, yp_tree, 'Tree');

fprintf('\n=== Part 8 Results (Test Set) ===\n');
modelsList = fieldnames(results);
bestName = modelsList{1}; bestR2 = results.(bestName).R2;
for i = 1:numel(modelsList)
    R = results.(modelsList{i});
    fprintf('%-20s  R^2 = %.3f | MAE = %.3f | RMSE = %.3f\n', ...
        modelsList{i}, R.R2, R.MAE, R.RMSE);
    if R.R2 > bestR2
        bestR2 = R.R2; bestName = modelsList{i};
    end
end

bestYp = results.(bestName).yp;
figure; scatter(Ytest, bestYp, 30, 'filled'); grid on; hold on;
plot([min(Ytest) max(Ytest)],[min(Ytest) max(Ytest)],'k--','LineWidth',1.5);
xlabel('True PWV_{cf} (m/s)'); ylabel('Predicted PWV_{cf} (m/s)');
title(sprintf('Part 8 — %s: Test True vs Pred', bestName));

fprintf('Part 8: Classical ML completed\n');
end

function R = evalReg(ytrue, ypred, tag)
ytrue = ytrue(:); ypred = ypred(:);
resid = ytrue - ypred;
SSres = sum(resid.^2);
SStot = sum( (ytrue - mean(ytrue)).^2 );
R.R2   = 1 - SSres/SStot;
R.MAE  = mean(abs(resid));
R.RMSE = sqrt(mean(resid.^2));
R.yp   = ypred;
R.tag  = tag;
end