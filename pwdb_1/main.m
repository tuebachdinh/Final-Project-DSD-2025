%% --- Part 1: Data Preparation and Wave Extraction ---
data = load('exported_data/pwdb_data.mat');
data = data.data; % Main struct

% Only keep physiologically plausible subjects
plausible = logical(data.plausibility.plausibility_log);
plausible = plausible(:); % Ensure column
N = sum(plausible); % 3837 plausible subjects

fs = data.waves.fs;
haemods = data.haemods;
plaus_idx = find(plausible);

age = [haemods(plaus_idx).age]';
PWV_cf = [haemods(plaus_idx).PWV_cf]';

% Define wave types and sites to analyze
wave_types = {'P', 'U', 'A', 'PPG'};
sites = {'AorticRoot', 'Radial', 'Brachial', 'Femoral', 'Digital'};

% Structure to hold all waves
waves = struct(); % Truncated version of data.waves for analysing
minLens = struct();

% Extract and truncate all waves for all types/sites
for w = 1:length(wave_types)
    for s = 1:length(sites)
        f = sprintf('%s_%s', wave_types{w}, sites{s});
        if isfield(data.waves, f)
            % Get cells for plausible subjects
            cellwaves = data.waves.(f)(plausible);
            lens = cellfun(@numel, cellwaves);
            minLens.(f) = min(lens);
            % Truncate and build [subjects x time]
            trunc = cellfun(@(x) x(1:minLens.(f)), cellwaves, 'UniformOutput', false);
            waves.(f) = cell2mat(trunc).';
        end
    end
end

% For plotting, pick Radial A (luminal area) as the main wave of interest
A_Radial = waves.A_Radial; % time-normalized to a single cardiac cycle/1 normalized heartbeat
t_A = (0:size(A_Radial,2)-1) / fs;

%% --- Part 2.1 Plot All Wave Types at Radial by Stiffness Group (PWV tertile) ---
% This helps visualize differences in P, U, A, PPG simultaneously by stiffness
edges_PWV = quantile(PWV_cf, [0 1/3 2/3 1]);
[~, bin_PWV] = histc(PWV_cf, edges_PWV);

labels_PWV = {'Low PWV','Medium PWV','High PWV'};
colors_PWV = lines(3); % 3 colors for 3 groups

figure('Position',[100 100 900 700]);
for i = 1:length(wave_types)
    subplot(2,2,i); hold on;
    for k = 1:3
        idx = (bin_PWV == k);
        f = sprintf('%s_Radial', wave_types{i});
        t = (0:size(waves.(f),2)-1)/fs;
        if any(idx)
            plot(t, mean(waves.(f)(idx,:),1), 'LineWidth',2, ...
                'Color', colors_PWV(k,:), 'DisplayName', labels_PWV{k});
        end
    end
    xlabel('Time (s)');
    ylabel(wave_types{i});
    title([wave_types{i} ' (Radial) by Stiffness Group']);
    legend('show');
    hold off;
end


%% --- Part 2.2: Plot All Wave Types at Radial by Age Group ---
% This helps visualize differences in P, U, A, PPG simultaneously
edges = [20 35 45 55 65 80]; % Group ages by decade (or set as you wish)
age_labels = arrayfun(@(a,b) sprintf('%d-%d',a,b-1), edges(1:end-1), edges(2:end),'uni',0);
[~, age_bin] = histc(age, edges);
colors = lines(length(edges)-1);

figure('Position',[100 100 900 700]);
for i = 1:length(wave_types)
    subplot(2,2,i); hold on;
    for k = 1:max(age_bin)
        idx = (age_bin == k);
        f = sprintf('%s_Radial', wave_types{i});
        t = (0:size(waves.(f),2)-1)/fs;
        if any(idx)
            plot(t, mean(waves.(f)(idx,:),1), 'LineWidth',2, 'Color', colors(k,:), 'DisplayName', age_labels{k});
        end
    end
    xlabel('Time (s)');
    ylabel(wave_types{i});
    title([wave_types{i} ' (Radial) by Age Group']);
    legend('show');
    hold off;
end

%% --- Part 2.3: Insights about time delay ---



%% --- Part 3: Age Dependence of Key Features from A_Radial ---

% Extract features: peak, mean, min, (add more as needed)
A_peak = max(A_Radial, [], 2);
A_mean = mean(A_Radial, 2);
A_min = min(A_Radial, [], 2);

% Plot A_peak vs Age (scatter)
figure;
scatter(age, A_peak, 30, 'b', 'filled'); grid on;
xlabel('Age (years)'); ylabel('Radial Area Peak (m^2)');
title('Peak Radial Area vs Age');

% Correlation
[rA, pA] = corr(age, A_peak);
fprintf('Correlation (peak A vs age): r = %.2f, p = %.3g\n', rA, pA);

%% --- Part 4: Feature Table and Regression (Age + Area) to PWV ---

T = table(age, A_peak, A_mean, A_min, PWV_cf, ...
    'VariableNames', {'Age','Apeak','Amean','Amin','PWV_cf'});

% Linear regression
mdl = fitlm(T, 'PWV_cf ~ Age + Apeak + Amean + Amin');
disp(mdl);

% Plot residuals
figure; plotResiduals(mdl,'fitted'); title('Residuals of PWV_cf Regression');

%% --- Part 5: Classical PPG Feature-based Tree Regression ---
RI     = [haemods(plaus_idx).RI]';
SI     = [haemods(plaus_idx).SI]';
AGImod = [haemods(plaus_idx).AGI_mod]';

% ML using only classical features
X = [RI, SI, AGImod];  % Only classical features
y = PWV_cf;
N = size(X,1);

cv = cvpartition(N, 'HoldOut', 0.2);
idxTrain = training(cv); idxTest = test(cv);

X_train = X(idxTrain,:);
y_train = y(idxTrain);
X_test = X(idxTest,:);
y_test = y(idxTest);

tree_feat = fitrtree(X_train, y_train);  % Tree regression on features
y_pred_feat = predict(tree_feat, X_test);

fprintf('Classical features tree, r = %.2f\n', corr(y_pred_feat, y_test));
figure;
scatter(y_test, y_pred_feat, 'filled');
hold on;
plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], 'k--', 'LineWidth', 2);
hold off;
xlabel('True PWV'); ylabel('Predicted PWV');
title('Test Set: Classical Features Tree Regression');
grid on;

%% --- Part 6: Optional: ML on Full Area Waveforms (+ Age as feature) ---

X_A = [A_Radial, age]; % Concatenate age as last feature column
y = PWV_cf;

N = size(X_A, 1);
cv = cvpartition(N, 'HoldOut', 0.2);
idxTrain = training(cv); idxTest = test(cv);
X_train = X_A(idxTrain,:);
y_train = y(idxTrain);
X_test = X_A(idxTest,:);
y_test = y(idxTest);

treeA = fitrtree(X_train, y_train);
y_predA = predict(treeA, X_test);

R_A = corr(y_predA, y_test);
fprintf('ML prediction, Area+Age: r = %.2f\n', R_A);

figure;
scatter(y_test, y_predA, 40, 'filled'); grid on;
hold on; plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], 'k--','LineWidth',2);
xlabel('True PWV'); ylabel('Predicted PWV');
title('Test Set: PWV Regression (A_{Radial} + Age, Tree Model)');

%% --- PART 7: Machine Learning on Full Wrist PPG Waveforms ---

% Input features: Each subject's full wrist PPG waveform
X = waves.PPG_Radial;     % size: [N_subjects x N_timepoints]

% Target variable: Central aortic PWV (stiffness)
y = PWV_cf;      % size: [N_subjects x 1]

N = size(X, 1);

% Split data into training and test sets (80/20 split)
cv = cvpartition(N, 'HoldOut', 0.2);
idxTrain = training(cv);
idxTest = test(cv);

X_train = X(idxTrain, :);
y_train = y(idxTrain);
X_test = X(idxTest, :);
y_test = y(idxTest);

% Train a regression tree model (can also use other models: SVM, ensemble, etc.)
tree = fitrtree(X_train, y_train);

% Predict on test set
y_pred = predict(tree, X_test);

% Evaluate model performance
R = corr(y_pred, y_test);   % Correlation coefficient
fprintf('Correlation (r) between predicted and true PWV: %.2f\n', R);

% Plot true vs. predicted PWV for the test set
figure;
scatter(y_test, y_pred, 'filled');
hold on;
plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], 'k--', 'LineWidth', 2); % 45-degree line
hold off;
xlabel('True PWV');
ylabel('Predicted PWV');
title('Test Set: Predicted vs. True PWV (Regression Tree)');
grid on;

%% --- End: Example Visualization of All Waves for One Subject ---
subject_id = 1;
figure('Position',[200 200 900 500]);
for i = 1:length(wave_types)
    subplot(2,2,i);
    f = sprintf('%s_Radial', wave_types{i});
    t = (0:size(waves.(f),2)-1)/fs;
    plot(t, waves.(f)(subject_id,:), 'LineWidth',2);
    xlabel('Time (s)'); ylabel(wave_types{i});
    title(sprintf('%s (Radial), Subject #%d, Age %d',wave_types{i},subject_id,age(subject_id)));
end

%% ---------------------------
% This script is now modular: you can easily add new sites, new wave types, new features, and advanced ML models as desired.
% For deeper analyses: try frequency domain features, time-to-peak, dicrotic notch, area under curve, or combine features from multiple wave types/sites.
