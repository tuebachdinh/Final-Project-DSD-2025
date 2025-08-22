function part4_feature_regression(data, plaus_idx, PWV_cf, haemods)
%PART4_FEATURE_REGRESSION Linear and tree regression with features

addpath('../utils/others');
pw_inds = data.pw_inds;

% Part 4.1: Linear regression with area features
age_feat = pw_inds.age(plaus_idx);
Amean_feat = pw_inds.Radial_Amean(plaus_idx);
Amin_feat = pw_inds.Radial_Amin(plaus_idx);
Amax_feat = pw_inds.Radial_Amax_V(plaus_idx);
PWV_feat = PWV_cf;

T = table(age_feat, Amax_feat, Amean_feat, Amin_feat, PWV_feat, ...
    'VariableNames', {'Age','Amax','Amean','Amin','PWV_cf'});

mdl = fitlm(T, 'PWV_cf ~ Age + Amax + Amean + Amin');
disp(mdl);

figure;
plotResiduals(mdl,'fitted');
title('Residuals of PWV_cf Regression');
save_figure('linear_regression_residuals', 4);

% Part 4.2: Tree regression with classical PPG feature-based
RI = [haemods(plaus_idx).RI]';
SI = [haemods(plaus_idx).SI]';
AGImod = [haemods(plaus_idx).AGI_mod]';

X = [RI, SI, AGImod];
y = PWV_cf;
N = size(X,1);

cv = cvpartition(N, 'HoldOut', 0.2);
idxTrain = training(cv);
idxTest = test(cv);

X_train = X(idxTrain,:);
y_train = y(idxTrain);
X_test = X(idxTest,:);
y_test = y(idxTest);

tree_feat = fitrtree(X_train, y_train);
y_pred_feat = predict(tree_feat, X_test);

fprintf('Classical features tree, r = %.2f\n', corr(y_pred_feat, y_test));

figure;
scatter(y_test, y_pred_feat, 'filled');
hold on;
plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], 'k--', 'LineWidth', 2);
xlabel('True PWV'); ylabel('Predicted PWV');
title('Test Set: Classical Features Tree Regression');
grid on;
save_figure('tree_regression_classical_features', 4);

fprintf('Part 4: Feature regression completed\n');
end