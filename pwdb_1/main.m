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



%% --- Part 2.1: Plot all wave types at chosen site with visible bands by PWV group ---
% Choose site to visualize
site = 'Radial';   % 'AorticRoot','Brachial','Femoral','Digital' also valid

% ----- PWV tertiles, labels, colors -----
pwv = PWV_cf(:);
valid = ~isnan(pwv);
q = quantile(pwv(valid), [1/3 2/3]);

bin_PWV = nan(size(pwv));
bin_PWV(pwv <= q(1))                 = 1;
bin_PWV(pwv > q(1) & pwv <= q(2))    = 2;
bin_PWV(pwv > q(2))                  = 3;

labels_PWV = {
    sprintf('Low PWV (%.1f–%.1f m/s)', min(pwv(valid)), q(1))
    sprintf('Medium PWV (%.1f–%.1f m/s)', q(1), q(2))
    sprintf('High PWV (%.1f–%.1f m/s)', q(2), max(pwv(valid)))
};
if ~exist('colors_PWV','var') || isempty(colors_PWV)
    colors_PWV = lines(3);  % 3 distinct colors
end

% ----- shading mode: choose one -----
shade_mode = 'sd';   % 'ci' (true 95% CI), 'sd' (±K·SD), or 'ci_x5' (scaled CI for visibility)
K = 0.5;               % only used if shade_mode = 'sd'

% ----- Plot -----
figure('Position',[100 100 900 700]);
for i = 1:numel(wave_types)
    subplot(2,2,i); cla; hold on;

    f = sprintf('%s_%s', wave_types{i}, site);
    if ~isfield(waves, f) || isempty(waves.(f))
        title(sprintf('%s (%s) missing', wave_types{i}, site));
        axis off; continue;
    end

    W = waves.(f);                       % [subjects x time]
    t = (0:size(W,2)-1)/fs;

    for k = 1:3
        idx = (bin_PWV == k);
        if ~any(idx), continue; end

        data_group = W(idx, :);
        n = sum(idx);
        mean_w = mean(data_group, 1, 'omitnan');
        std_w  = std(data_group, 0, 1, 'omitnan');

        % ---- band width selection ----
        switch shade_mode
            case 'ci'
                band = 1.96 * (std_w ./ sqrt(n));
                band_note = '95% CI';
            case 'sd'
                band = K * std_w;
                band_note = sprintf('±%.1fσ band', K);
            case 'ci_x5'
                band = 5 * 1.96 * (std_w ./ sqrt(n));
                band_note = '95% CI (×5 for visibility)';
            otherwise
                band = 1.96 * (std_w ./ sqrt(n));
                band_note = '95% CI';
        end

        % % ---- shaded band (kept out of legend) ----
        fill([t, fliplr(t)], [mean_w+band, fliplr(mean_w-band)], ...
            colors_PWV(k,:), 'FaceAlpha',0.20, 'EdgeColor','none', ...
            'HandleVisibility','off');

        % optional boundary lines (also hidden from legend)
        plot(t, mean_w+band, ':', 'Color', colors_PWV(k,:), 'HandleVisibility','off');
        plot(t, mean_w-band, ':', 'Color', colors_PWV(k,:), 'HandleVisibility','off');

        % ---- mean line (legend item) ----
        plot(t, mean_w, 'LineWidth', 3, 'Color', colors_PWV(k,:), ...
            'DisplayName', sprintf('%s', labels_PWV{k}));
    end

    xlabel('Time (s)');
    switch wave_types{i}
        case 'P',   ylabel('Pressure (mmHg)');
        case 'U',   ylabel('Flow velocity (m/s)');
        case 'A',   ylabel('Luminal area (m^3)');   % consider rescaling to ×10^{-6} if tiny
        case 'PPG', ylabel('PPG (a.u.)');
    end
    title(sprintf('%s (%s) by Stiffness Group', wave_types{i}, site));
    grid on; box on;
    axis tight;
    yl = ylim; ylim(yl + [-1 1]*0.05*range(yl));   % small padding
    legend('Location','best');
end
sgtitle(sprintf('Waveforms at %s with %s by PWV Group', site, band_note));  % sgtitle can't render × in some fonts



%% --- Part 2.2: Plot All Wave Types at Radial by Age Group ---
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
    switch wave_types{i}
        case 'P',   ylabel('Pressure (mmHg)');
        case 'U',   ylabel('Flow velocity (m/s)');
        case 'A',   ylabel('Luminal area (m^3)');   % consider rescaling to ×10^{-6} if tiny
        case 'PPG', ylabel('PPG (a.u.)');
    end
    title(sprintf('%s (%s) by Age Group', wave_types{i}, site));
    grid on; box on;
    axis tight;
    yl = ylim; ylim(yl + [-1 1]*0.05*range(yl));   % small padding
    legend('Location','best');
    hold off;
end
%sgtitle(sprintf('Waveforms at %s by Age Group', site))


%% --- Part 2.3: Insights about time delay ---
% Using the onset times, and manually calculate the PTT
% Get pulse onset times (in seconds) for each site, each subject
onsets = data.waves.onset_times;
onset_aortic = onsets.P_AorticRoot(plaus_idx); % [Nsubjects x 1], in sec
onset_radial = onsets.P_Radial(plaus_idx);     % [Nsubjects x 1], in sec
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

%% Box Plot: PTT vs Age
unique_ages = unique(age); % should be [25 35 45 55 65 75]
figure;
boxplot(PTT_aor_to_rad, age, ...
    'Labels', string(unique_ages));

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
subject_idx = 6; % Or any plausible subject index
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

%% Part 5.5: Using New PulseAnalyse on Same Radial PPG Data

% Use the same signal from Part 5.1
S_new = struct();
S_new.v = signal(:);  % Same signal as Part 5.1
S_new.fs = fs;
% S_new.ht = 1.75;    % Optional: subject height in meters

% Configure options for new PulseAnalyse
options_new = struct();
options_new.do_plot = true;
options_new.do_filter = 1;
options_new.do_quality = 1;
options_new.normalise_pw = 1;
options_new.beat_detector = 'IMS';

% Run the new pulse analysis function
[pw_inds_new, fid_pts_new, pulses_new, sigs_new] = PulseAnalyse(S_new, options_new);

% Display key results
fprintf('\n=== New PulseAnalyse Results ===\n');
if ~isempty(pw_inds_new)
    fprintf('Augmentation Index (AI): %.2f%%\n', pw_inds_new.AI.v);
    fprintf('Reflection Index (RI): %.2f\n', pw_inds_new.RI.v);
    fprintf('Stiffness Index (SI): %.2f m/s\n', pw_inds_new.SI.v);
    fprintf('Crest Time (CT): %.3f s\n', pw_inds_new.CT.v);
    fprintf('Delta T: %.3f s\n', pw_inds_new.delta_t.v);
end

% Compare fiducial points between old and new versions
figure('Position', [100 100 1200 400]);
subplot(1,2,1);
plot(S.v, 'b', 'LineWidth', 1.5); hold on;
if exist('fid_pts', 'var') && isfield(fid_pts, 'dic')
    plot(fid_pts.dic, S.v(fid_pts.dic), 'ro', 'MarkerSize', 8);
    plot(fid_pts.p2pk, S.v(fid_pts.p2pk), 'go', 'MarkerSize', 8);
    plot(fid_pts.dia, S_new.v(fid_pts.dia), 'mo', 'MarkerSize', 8);

end
title('PulseAnalyse10 (Old Version)');
xlabel('Sample'); ylabel('Amplitude');
legend('PPG Signal', 'Dicrotic Notch', 'Second Peak', 'Diastolic Peak');


subplot(1,2,2);
plot(S_new.v, 'b', 'LineWidth', 1.5); hold on;
if ~isempty(fid_pts_new) && ~isnan(fid_pts_new.ind.dic)
    plot(fid_pts_new.ind.dic, S_new.v(fid_pts_new.ind.dic), 'ro', 'MarkerSize', 8);
    plot(fid_pts_new.ind.p2pk, S_new.v(fid_pts_new.ind.p2pk), 'go', 'MarkerSize', 8);
    plot(fid_pts_new.ind.dia, S_new.v(fid_pts_new.ind.dia), 'mo', 'MarkerSize', 8);
end
title('PulseAnalyse (New Version)');
xlabel('Sample'); ylabel('Amplitude');
legend('PPG Signal', 'Dicrotic Notch', 'Second Peak', 'Diastolic Peak');
sgtitle('Comparison: Old vs New PulseAnalyse');




%% --- Part 6: Feature-based Machine Learning (PPG + Area + Timing) ---
% Goal: Build a robust feature set (cheap to compute), train light models,
% and evaluate R^2 / RMSE / MAE. Save the best model for later transfer.

rng(42);  % reproducibility

% ---------- 6.1 Assemble features per subject ----------
Nsubj = numel(plaus_idx);

% -- PPG classical (from haemods) --
RI     = [haemods(plaus_idx).RI]';
SI     = [haemods(plaus_idx).SI]';
AGImod = [haemods(plaus_idx).AGI_mod]';

% -- PPG sdPPG & morphology (from pw_inds) --
PPGa   = pw_inds.Radial_PPGa_V(plaus_idx);
PPGb   = pw_inds.Radial_PPGb_V(plaus_idx);
PPGc   = pw_inds.Radial_PPGc_V(plaus_idx);
PPGd   = pw_inds.Radial_PPGd_V(plaus_idx);
PPGe   = pw_inds.Radial_PPGe_V(plaus_idx);
PPGsys = pw_inds.Radial_PPGsys_V(plaus_idx);
PPGdia = pw_inds.Radial_PPGdia_V(plaus_idx);
PPGdic = pw_inds.Radial_PPGdic_V(plaus_idx);
PPGms  = pw_inds.Radial_PPGms_V(plaus_idx);

% -- Area (A) features at wrist (Radial) --
Amax   = pw_inds.Radial_Amax_V(plaus_idx);   
Amin   = pw_inds.Radial_Amin(plaus_idx);
Amean  = pw_inds.Radial_Amean(plaus_idx);
Astk   = Amax - Amin;                                % stroke area change
Aosc   = (Amax - Amin) ./ max(Amean, eps);           % relative oscillation 


% -- Timing features --
if ~exist('PTT_aor_to_rad','var') || isempty(PTT_aor_to_rad)
    onset_aortic = data.waves.onset_times.P_AorticRoot(plaus_idx);
    onset_radial = data.waves.onset_times.P_Radial(plaus_idx);
    PTT_aor_to_rad = onset_radial - onset_aortic;    % secs
end
HR = [haemods(plaus_idx).HR]';                        % bpm
LVET = [haemods(plaus_idx).LVET]';                    % ms (proxy for systole dur)
PFT  = [haemods(plaus_idx).PFT]';                     % ms (time of peak aortic flow)

% -- Coupling features between PPG & Area (wrist) --
PPG_R = waves.PPG_Radial;     % [N x T]
A_R   = waves.A_Radial;       % [N x T]
Tpts  = size(PPG_R,2);
tgrid = (0:Tpts-1)./fs;

% peak timings (in seconds)
[~, ppg_pk_idx] = max(PPG_R, [], 2);
[~,  A_pk_idx ] = max(A_R,   [], 2);
t_ppg_pk = tgrid(ppg_pk_idx)';
t_A_pk   = tgrid(A_pk_idx)';
lag_A_vs_PPG = t_A_pk - t_ppg_pk;             % +ve: A peaks after PPG

% simple cross-correlation over systole (use first half of the beat)
mid_idx = round(Tpts*0.6);
xcorr_pp = nan(Nsubj,1);
for ii = 1:Nsubj
    x1 = detrend(PPG_R(ii,1:mid_idx));
    x2 = detrend(A_R(ii,1:mid_idx));
    c  = corr(x1(:), x2(:), 'rows','complete');
    xcorr_pp(ii) = c;
end

% -- Simple peak detection features on PPG (cheap & robust) --
% Find primary and secondary peaks using simple methods
P1samp = nan(Nsubj,1); P2samp = nan(Nsubj,1);
P1_amp = nan(Nsubj,1); P2_amp = nan(Nsubj,1);
for ii = 1:Nsubj
    x = PPG_R(ii,:)';
    % min-max normalize
    xx = x; 
    rngv = max(xx)-min(xx);
    if rngv > 0, xx = (xx - min(xx)) / rngv; end
    
    % Find primary peak (systolic)
    [P1_amp(ii), P1samp(ii)] = max(xx);
    
    % Find secondary peak (dicrotic) in second half
    mid_idx = round(length(xx)/2);
    if P1samp(ii) < mid_idx
        [P2_amp(ii), p2_rel] = max(xx(P1samp(ii)+10:end));
        P2samp(ii) = P1samp(ii) + 10 + p2_rel - 1;
    else
        P2_amp(ii) = 0;
        P2samp(ii) = length(xx);
    end
end
% Convert P1/P2 sample indices to sec (relative to beat start)
tP1 = P1samp./fs;  tP2 = P2samp./fs;  dTPeaks = tP2 - tP1;

% -- Build feature table --
y = PWV_cf(:);                               % target (m/s)
Xtbl = table( ...
    age(:), HR(:), LVET(:), PFT(:), ...
    RI(:), SI(:), AGImod(:), ...
    PPGa(:), PPGb(:), PPGc(:), PPGd(:), PPGe(:), ...
    PPGsys(:), PPGdia(:), PPGdic(:), PPGms(:), ...
    Amax(:), Amin(:), Amean(:), Astk(:), Aosc(:), ...
    PTT_aor_to_rad(:), ...
    t_ppg_pk(:), t_A_pk(:), lag_A_vs_PPG(:), xcorr_pp(:), ...
    P1_amp, P2_amp, tP1, tP2, dTPeaks, ...
    y, ...
    'VariableNames', { ...
    'Age','HR','LVET','PFT', ...
    'RI','SI','AGImod', ...
    'PPGa','PPGb','PPGc','PPGd','PPGe', ...
    'PPGsys','PPGdia','PPGdic','PPGms', ...
    'Amax','Amin','Amean','Astk','Aosc', ...
    'PTT_hw', ...
    'tPPGpk','tApk','lagAminusPPG','xcorrSyst', ...
    'P1_amp','P2_amp','tP1','tP2','dTPeaks', ...
    'PWV_cf'});

% Clean NaNs/Infs
bad = any(~isfinite(Xtbl{:,:}), 2);
Xtbl = Xtbl(~bad, :);

% ---------- 6.2 Train/test split ----------
N = height(Xtbl);
cv = cvpartition(N, 'HoldOut', 0.2);
Xnames = Xtbl.Properties.VariableNames;
Xnames(end) = [];                     % remove PWV_cf from predictors
Yname = 'PWV_cf';
Xtrain = Xtbl(training(cv), Xnames);
Ytrain = Xtbl{training(cv), Yname};
Xtest  = Xtbl(test(cv), Xnames);
Ytest  = Xtbl{test(cv),  Yname};

% z-score standardize numeric predictors (store params for deployment)
[Ztrain, muX, stdX] = zscore(table2array(Xtrain));
Ztest = (table2array(Xtest) - muX) ./ stdX;

% ---------- 6.3 Models (light & deployable) ----------
results = struct();

% (a) Ridge Regression
ridgeMdl = fitrlinear(Ztrain, Ytrain, ...
    'Learner','leastsquares', 'Regularization','ridge', ...
    'KFold',5, 'Solver','lbfgs');
ridgeTrained = fitrlinear(Ztrain, Ytrain, ...
    'Learner','leastsquares', 'Regularization','ridge', ...
    'Solver','lbfgs');
yp_ridge = predict(ridgeTrained, Ztest);
results.Ridge = evalReg(Ytest, yp_ridge, 'Ridge');

% (b) Lasso (with fixed lambda)
lassoTrained = fitrlinear(Ztrain, Ytrain, ...
    'Learner','leastsquares','Regularization','lasso', ...
    'Lambda', 0.01, 'Solver','sparsa');
yp_lasso = predict(lassoTrained, Ztest);
results.Lasso = evalReg(Ytest, yp_lasso, 'Lasso');

% (c) Partial Least Squares (PLS) with small latent dims
maxComps = min(10, size(Ztrain,2));
[Rpls,~,~,~,beta, PCTVAR] = plsregress(Ztrain, Ytrain, min(6,maxComps));
yp_pls = [ones(size(Ztest,1),1) Ztest]*beta;
results.PLS = evalReg(Ytest, yp_pls, sprintf('PLS (%d comps)', size(beta,1)-1));

% (d) Small Tree
treeMdl = fitrtree(Ztrain, Ytrain, 'MinLeafSize', 100, 'MaxNumSplits', 10);
yp_tree = predict(treeMdl, Ztest);
results.Tree = evalReg(Ytest, yp_tree, 'Tree (small)');

% (e) Shallow Ensemble (LSBoost) — keep tiny to avoid overfit
treeTemplate = templateTree('MinLeafSize', 80, 'MaxNumSplits', 20);
ensMdl = fitrensemble(Ztrain, Ytrain, 'Method','LSBoost', ...
    'NumLearningCycles', 60, 'LearnRate', 0.05, ...
    'Learners', treeTemplate);
yp_ens = predict(ensMdl, Ztest);
results.Ensemble = evalReg(Ytest, yp_ens, 'Ensemble (shallow)');

% ---------- 6.4 Compare and report ----------
fprintf('\n=== Part 6 Results (Test Set) ===\n');
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
fprintf('=> Best (by R^2): %s\n', bestName);

% Plot True vs Pred for best
bestYp = results.(bestName).yp;
figure; scatter(Ytest, bestYp, 30, 'filled'); grid on; hold on;
plot([min(Ytest) max(Ytest)],[min(Ytest) max(Ytest)],'k--','LineWidth',1.5);
xlabel('True PWV_{cf} (m/s)'); ylabel('Predicted PWV_{cf} (m/s)');
title(sprintf('Part 6 — %s: Test True vs Pred', bestName));

% ---------- 6.5 Save best model & normalization for deployment ----------
BestModelPackage = struct('name', bestName, 'muX', muX, 'stdX', stdX, ...
    'Xnames', {Xnames}, 'target', Yname);
switch bestName
    case 'Ridge'
        BestModelPackage.model = ridgeTrained;
    case 'Lasso'
        BestModelPackage.model = lassoTrained;
    case 'PLS'
        BestModelPackage.model = struct('beta', beta, 'intercept', beta(1), ...
            'pctvarX', PCTVAR(1,:), 'pctvarY', PCTVAR(2,:));
    case 'Tree'
        BestModelPackage.model = treeMdl;
    case 'Ensemble'
        BestModelPackage.model = ensMdl;
end
save('part6_best_model.mat','BestModelPackage');

% ---------- 6.6 Helper functions ----------
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





%% --- Part 7: Simple Deep Learning (Single Input CNN) ---
% Simplified version using concatenated features instead of multi-input

rng(7);  % reproducibility

% ---------- 7.1 Prepare data ----------
X_ppg = waves.PPG_Radial;   % [N x T]
X_area = waves.A_Radial;    % [N x T]
y = PWV_cf(:);

% Filter out rows with NaNs/Infs or missing targets
good = all(isfinite(X_ppg),2) & all(isfinite(X_area),2) & isfinite(y);
X_ppg = X_ppg(good,:); 
X_area = X_area(good,:);
y = y(good);

N = size(X_ppg,1);
T = size(X_ppg,2);

% Normalize each signal
X_ppg = (X_ppg - mean(X_ppg,2)) ./ max(std(X_ppg,[],2), eps);
X_area = (X_area - mean(X_area,2)) ./ max(std(X_area,[],2), eps);

% Concatenate PPG and Area as 2-channel input
X_combined = cat(3, X_ppg, X_area);  % [N x T x 2]
X_combined = permute(X_combined, [3 2 1]);  % [2 x T x N] for MATLAB

% Convert to cell array of sequences
seqData = cell(N,1);
for i = 1:N
    seqData{i} = X_combined(:,:,i);  % [2 x T]
end

% ---------- 7.2 Train/Test split ----------
idx = randperm(N);
nTrain = round(0.8*N);
trainIdx = idx(1:nTrain);
testIdx = idx(nTrain+1:end);

% ---------- 7.3 Simple 1D CNN ----------
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

% ---------- 7.4 Training options ----------
opts = trainingOptions('adam', ...
    'InitialLearnRate', 1e-3, ...
    'MaxEpochs', 40, ...
    'MiniBatchSize', 32, ...
    'Shuffle', 'every-epoch', ...
    'Verbose', false);

% ---------- 7.5 Train ----------
net = trainNetwork(seqData(trainIdx), y(trainIdx), layers, opts);

% ---------- 7.6 Evaluate ----------
yp = predict(net, seqData(testIdx));
ytrue = y(testIdx);

% Metrics
resid = ytrue - yp;
R2 = 1 - sum(resid.^2) / sum((ytrue - mean(ytrue)).^2);
MAE = mean(abs(resid));
RMSE = sqrt(mean(resid.^2));

fprintf('\n=== Part 7 (Simple CNN) — Test Metrics ===\n');
fprintf('R^2 = %.3f | MAE = %.3f m/s | RMSE = %.3f m/s\n', R2, MAE, RMSE);

% Plot
figure;
scatter(ytrue, yp, 30, 'filled'); grid on; hold on;
plot([min(ytrue) max(ytrue)], [min(ytrue) max(ytrue)], 'k--', 'LineWidth', 1.5);
xlabel('True PWV_{cf} (m/s)'); ylabel('Predicted PWV_{cf} (m/s)');
title('Part 7 — Simple CNN: Test True vs Pred');

% Save model
save('part7_simple_cnn.mat', 'net');