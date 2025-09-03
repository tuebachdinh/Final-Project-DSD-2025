function part5_ml_waveforms(waves, PWV_cf, age, fs)
%PART5_ML_WAVEFORMS Machine Learning using Time Series on Full Waveforms

addpath('../utils/others');

% Part 5.1: Area waveforms
X_A = waves.A_Radial;
y = PWV_cf;

N = size(X_A, 1);
cv = cvpartition(N, 'HoldOut', 0.2);
idxTrain = training(cv);
idxTest = test(cv);

X_train = X_A(idxTrain,:);
y_train = y(idxTrain);
X_test = X_A(idxTest,:);
y_test = y(idxTest);

treeA = fitrtree(X_train, y_train);
y_predA = predict(treeA, X_test);

R_A = corr(y_predA, y_test);
fprintf('ML prediction, Area Radial time series: r = %.2f\n', R_A);

figure;
scatter(y_test, y_predA, 40, 'filled'); grid on;
hold on; plot([min(y_test) max(y_test)], [min(y_test) max(y_test)], 'k--','LineWidth',2);
xlabel('True PWV'); ylabel('Predicted PWV');
title('Test Set: PWV Regression (A_{Radial}, Tree Model)');
save_figure('area_waveform_regression', 5);

% Part 5.2: PPG waveforms
X = waves.PPG_Radial;
y = PWV_cf;

N = size(X, 1);
cv = cvpartition(N, 'HoldOut', 0.2);
idxTrain = training(cv);
idxTest = test(cv);

X_train = X(idxTrain, :);
y_train = y(idxTrain);
X_test = X(idxTest, :);
y_test = y(idxTest);

tree = fitrtree(X_train, y_train);
y_pred = predict(tree, X_test);

R = corr(y_pred, y_test);
fprintf('ML prediction, PPG Radial time series: %.2f\n', R);

figure;
scatter(y_test, y_pred, 'filled');
hold on;
plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], 'k--', 'LineWidth', 2);
xlabel('True PWV'); ylabel('Predicted PWV');
title('Test Set: Predicted vs. True PWV (Regression Tree)');
grid on;
save_figure('ppg_waveform_regression', 5);

% Example visualization
site = 'Radial';
subject_id = 2695;
figure('Position',[200 200 900 500]);
wave_types = {'P', 'U', 'A', 'PPG'};
for i = 1:length(wave_types)
    subplot(2,2,i);
    f = sprintf('%s_%s', wave_types{i}, site);
    t = (0:size(waves.(f),2)-1)/fs;
    plot(t, waves.(f)(subject_id,:), 'LineWidth',2);
    xlabel('Time (s)'); ylabel(wave_types{i});
    title(sprintf('%s (%s), Subject #%d',wave_types{i},site, subject_id));
end
save_figure('example_subject_waveforms', 5);

fprintf('Part 5: ML waveforms analysis completed\n');
end