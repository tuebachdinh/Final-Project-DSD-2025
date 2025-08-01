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

%Other paths example
PWV_ba = [haemods(plaus_idx).PWV_ba]';

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

site = 'Radial'; % or 'Brachial', 'AorticRoot', 'Femoral', 'Digital', etc.

edges_PWV = quantile(PWV_cf, [0 1/3 2/3 1]);
[~, bin_PWV] = histc(PWV_cf, edges_PWV);

labels_PWV = {'Low PWV','Medium PWV','High PWV'};
colors_PWV = lines(3); % 3 colors for 3 groups

figure('Position',[100 100 900 700]);
for i = 1:length(wave_types)
    subplot(2,2,i); hold on;
    for k = 1:3
        idx = (bin_PWV == k);
        f = sprintf('%s_%s', wave_types{i}, site);
        t = (0:size(waves.(f),2)-1)/fs;
        if any(idx)
            plot(t, mean(waves.(f)(idx,:),1), 'LineWidth',2, ...
                'Color', colors_PWV(k,:), 'DisplayName', labels_PWV{k});
        end
    end
    xlabel('Time (s)');
    ylabel(wave_types{i});
    title(sprintf('%s (%s) by Stiffness Group', wave_types{i}, site));
    legend('show');
    hold off;
end

%% --- Part 2.2: 4x3 Plot: Each Wave Type (row) by PWV Group (col), with mean ± std ---

site = 'Radial'; % Change as needed

edges_PWV = quantile(PWV_cf, [0 1/3 2/3 1]);
[~, bin_PWV] = histc(PWV_cf, edges_PWV);

labels_PWV = {'Low PWV','Medium PWV','High PWV'};

figure('Position',[100 100 1600 800]);
for i = 1:length(wave_types)
    f = sprintf('%s_%s', wave_types{i}, site);
    mat = waves.(f);
    t = (0:size(mat,2)-1)/fs;
    for k = 1:3
        idx = (bin_PWV == k);
        data_group = mat(idx,:);
        mean_w = mean(data_group,1);
        std_w = std(data_group,0,1);

        subplot(length(wave_types),3,(i-1)*3 + k); hold on;
        % Shaded area: mean ± std
        fill([t, fliplr(t)], ...
             [mean_w+std_w, fliplr(mean_w-std_w)], ...
             [0.6 0.6 0.9], 'FaceAlpha',0.3, 'EdgeColor','none');
        plot(t, mean_w, 'b', 'LineWidth',2);
        xlabel('Time (s)');
        ylabel(wave_types{i});
        title(sprintf('%s - %s', wave_types{i}, labels_PWV{k}));
        grid on;
        hold off;
    end
end
sgtitle(sprintf('Mean ± 1SD for Each Wave Type at %s by PWV Group', site));



%% --- Part 2.3: Plot All Wave Types at Radial by Age Group ---
% This helps visualize differences in P, U, A, PPG simultaneously

site = 'Radial'; % or 'Brachial', 'AorticRoot', 'Femoral', 'Digital', etc.

unique_ages = [25 35 45 55 65 75];
age_labels = arrayfun(@(a) sprintf('%d', a), unique_ages, 'uni', 0);
colors = lines(length(unique_ages));

figure('Position',[100 100 900 700]);
for i = 1:length(wave_types)
    subplot(2,2,i); hold on;
    for k = 1:length(unique_ages)
        idx = (age == unique_ages(k));
        f = sprintf('%s_%s', wave_types{i}, site);
        t = (0:size(waves.(f),2)-1)/fs;
        if any(idx)
            plot(t, mean(waves.(f)(idx,:),1), ...
                 'LineWidth',2, 'Color', colors(k,:), ...
                 'DisplayName', age_labels{k});
        end
    end
    xlabel('Time (s)');
    ylabel(wave_types{i});
    title(sprintf('%s (%s) by Age Group', wave_types{i}, site));
    legend('show');
    hold off;
end


%% --- Part 2.4: Insights about time delay ---

% Using the onset times, and manually calculate the PTT
% Get pulse onset times (in seconds) for each site, each subject
onsets = data.waves.onset_times;

% Get onset times at aortic root and at wrist (radial)
onset_aortic = onsets.P_AorticRoot(plaus_idx); % [Nsubjects x 1], in sec
onset_radial = onsets.P_Radial(plaus_idx);     % [Nsubjects x 1], in sec

% Pulse transit time: heart (aortic root) to wrist (radial)
PTT_aor_to_rad = onset_radial - onset_aortic;  % [Nsubjects x 1], in sec
fprintf('Mean PTT (aortic root to radial): %.3f s (std: %.3f s)\n', ...
    mean(PTT_aor_to_rad), std(PTT_aor_to_rad));


% Using the pw_inds for precalculated features (more complex signal processing)
pw_inds = data.pw_inds;
PTT_pwinds = pw_inds.Radial_PTT(plaus_idx);
PTT_pwinds_positive = PTT_pwinds(PTT_pwinds >= 0);
fprintf('PTT (from pw_inds, aortic root to radial, positive only): mean %.3f s (N = %d)\n', ...
    mean(PTT_pwinds_positive), numel(PTT_pwinds_positive));

figure;
histogram(PTT_pwinds_positive, 40);
xlabel('PTT (AorticRoot \rightarrow Radial) [s]');
ylabel('Count');
title('Distribution of Pulse Transit Time (PTT) from Heart to Wrist');
grid on;

%% Scatter Plot: PTT vs Age
figure;
scatter(age, PTT_aor_to_rad, 10, 'filled');
xlabel('Age (years)');
ylabel('PTT (s)');
title('PTT (Heart to Wrist) vs Age');
grid on;

%% --- Part 3.1: Precalculated Feature (Age + Area) and Linear Regression for PWV ---
% Always select plausible rows for every variable
age_feat   = pw_inds.age(plaus_idx);                 % Age is global (not per site)
Amean_feat = pw_inds.Radial_Amean(plaus_idx);
Amin_feat  = pw_inds.Radial_Amin(plaus_idx);
Amax_feat  = pw_inds.Radial_Amax_V(plaus_idx);         % Use _Amax not _Amax_V, unless only _Amax_V exists
PWV_feat   = PWV_cf;                                 % Already filtered

% Combine into table
T = table(age_feat, Amax_feat, Amean_feat, Amin_feat, PWV_feat, ...
    'VariableNames', {'Age','Amax','Amean','Amin','PWV_cf'});

% Linear regression
mdl = fitlm(T, 'PWV_cf ~ Age + Amax + Amean + Amin');
disp(mdl);

% Plot residuals
figure;
plotResiduals(mdl,'fitted'); % Actual PWV_cf − Predicted PWV_cf
title('Residuals of PWV_cf Regression');


%% --- Part 3.2: Classical PPG Feature-based Tree Regression ---
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

%% --- Part 4.1: Machine Learning using Time Series on Full Area Waveforms  ---

X_A = A_Radial; % Concatenate age as last feature column
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
fprintf('ML prediction, Area Radial time series: r = %.2f\n', R_A);

figure;
scatter(y_test, y_predA, 40, 'filled'); grid on;
hold on; plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], 'k--','LineWidth',2);
xlabel('True PWV'); ylabel('Predicted PWV');
title('Test Set: PWV Regression (A_{Radial}, Tree Model)');

%% --- Part 4.2: Machine Learning using Time Series on Full Wrist PPG Waveforms ---

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
fprintf('ML prediction, PPG Radial time series: %.2f\n', R);

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

%% --- Example Visualization of All Waves for One Subject ---
site = 'Radial'; % 'Radial', 'Brachial', 'AorticRoot', 'Femoral', 'Digital', etc.
subject_id = 6;
figure('Position',[200 200 900 500]);
for i = 1:length(wave_types)
    subplot(2,2,i);
    f = sprintf('%s_%s', wave_types{i}, site);
    t = (0:size(waves.(f),2)-1)/fs;
    plot(t, waves.(f)(subject_id,:), 'LineWidth',2);
    xlabel('Time (s)'); ylabel(wave_types{i});
    title(sprintf('%s (%s), Subject #%d, Age %d',wave_types{i},site, subject_id,age(subject_id)));
end

%% Part 5: Use of Signal Processing Algorithms to extract features, indexes from PW
addpath('/Users/tueeee/MATLAB-Drive/Final-Project-DSD-2025/algorithms/');

% --- Choose a subject and extract their PPG waveform ---
subject_idx = 1; % Or any plausible subject index
signal = waves.PPG_Radial(subject_idx, :);

% --- Prepare the structure for analysis ---
S = struct();
S.v = signal(:); % Column vector
S.fs = fs;
% S.ht = ...; % (Optional) subject's height in meters

%% Part 5.1: Using PulseAnalyse10 on a Single Radial PPG Beat to extract fiducial points and calculated indexes

options = struct();
options.do_plot = true; % Enable plotting

% --- Run the pulse analysis function ---
[cv_inds, fid_pts, pulses, S_filt] = PulseAnalyse10(S, options);

% --- Access key outputs ---
disp(cv_inds);     % Show calculated indices
disp(fid_pts);     % Show detected fiducial points
disp(pulses);      % Show pulse/beat info

% Example: Find and plot the dicrotic notch index
dic_idx = fid_pts.dic;
figure; plot(S.v);
hold on; plot(dic_idx, S.v(dic_idx), 'ro', 'MarkerSize', 8, 'DisplayName', 'Dicrotic Notch');
legend('Signal','Dicrotic Notch');
title('PPG with Detected Dicrotic Notch');
xlabel('Sample'); ylabel('Amplitude');
hold off;

%% Part 5.2: Gaussian Fitting for Second Peak (P2) Detection

[fitCurve, params, p1_idx, p2_idx] = fitTwoGaussiansPPG(S.v);

figure;
plot(S.v, 'b', 'DisplayName', 'Original Signal'); hold on;
plot(fitCurve, 'r--', 'DisplayName', 'Fitted 2-Gaussian');
plot([p1_idx, p2_idx], fitCurve([p1_idx, p2_idx]), 'ko', 'MarkerFaceColor','g','MarkerSize',10);
legend('show');
title('Gaussian Fitting: P1 & P2 Detection');
xlabel('Sample'); ylabel('Amplitude');

% (You can use p2_idx as an estimate for the second peak location)

%% Part 5.3: Frequency and Morphology Features
features = extractFreqMorphFeatures(S.v, S.fs);
disp('Frequency and Morphology Features:');
disp(features);

%% Part 5.4 Signal Quality Index (SQI) & Artifact Rejection
% (Assume you have a template - e.g., average of several good beats)
if exist('beats', 'var') % If you have a matrix of beats
    template = mean(beats,1);
else
    template = S.v; % Fallback: use current beat as its own template
end

[sqi, isGood] = computeSimpleSQI(S.v, template);

fprintf('SQI value: %.2f | Is Good: %d\n', sqi, isGood);

if ~isGood
    warning('This pulse may be of low quality or corrupted!');
end


