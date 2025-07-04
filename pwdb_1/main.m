%% --- Part 1: Data preparation ---
data = load('exported_data/pwdb_data.mat');
data = data.data; % Main struct

% Only keep physiologically plausible subjects
plausible = logical(data.plausibility.plausibility_log);
plausible = plausible(:); % Ensure column

% --- Get PPG at the wrist (radial artery) ---
PPG_cells = data.waves.PPG_Radial(plausible); % Cell array, length Nplausible

% Ensure all waveforms are the same length (truncate to minimum length)
waveLens = cellfun(@numel, PPG_cells);
minLen = min(waveLens);

% Truncate all to the same minimum length (so each cell is a column vector)
PPG_trunc = cellfun(@(x) x(1:minLen), PPG_cells, 'UniformOutput', false);

% Build PPG matrix: [subjects x time]
PPG_mat = cell2mat(PPG_trunc); % This makes [minLen x Nsubjects]
PPG_mat = PPG_mat.';           % Transpose: [Nsubjects x minLen]

fs = data.waves.fs; % Sampling frequency

%% --- Part 2.1: Waveform Analysis ---
haemods = data.haemods;

% Indices of plausible subjects
plaus_idx = find(plausible);

% Extract desired variables for plausible subjects
PWV_cf = [haemods(plaus_idx).PWV_cf]'; % as column
age    = [haemods(plaus_idx).age]';
HR     = [haemods(plaus_idx).HR]';

% --- Bin PWV into tertiles for plotting ---
edges = quantile(PWV_cf, [0 1/3 2/3 1]);
[~, bin] = histc(PWV_cf, edges);

% Time vector (seconds)
t = (0:minLen-1)/fs;

% --- Plot average PPG for low/medium/high stiffness ---
figure;
hold on;
colors = {'b','g','r'};
labels = {'Low PWV','Medium PWV','High PWV'};
for k = 1:3
    groupIdx = (bin == k);
    disp(['Group ', labels{k}, ' has ', num2str(sum(groupIdx)), ' subjects']);
    if any(groupIdx)
        meanPPG = mean(PPG_mat(groupIdx,:), 1);
        plot(t, meanPPG, 'Color', colors{k}, 'LineWidth', 2);
    end
end
legend(labels); xlabel('Time (s)'); ylabel('PPG (a.u.)');
title('Average Wrist PPG by Stiffness Group');
hold off;

%% --- Part 2.2: Pressure Waveform Analysis for Pressure ---
% --- Get Radial Pressure at the wrist (radial artery) ---
P_cells = data.waves.P_Radial(plausible); % Cell array, length Nplausible

waveLens_P = cellfun(@numel, P_cells);
minLen_P = min(waveLens_P);

P_trunc = cellfun(@(x) x(1:minLen_P), P_cells, 'UniformOutput', false);

% Build Pressure matrix: [subjects x time]
P_mat = cell2mat(P_trunc); % [minLen_P x Nsubjects]
P_mat = P_mat.';           % Transpose: [Nsubjects x minLen_P]
fs = data.waves.fs; % Sampling frequency (already available from above)

% Time vector (seconds)
t_P = (0:minLen_P-1)/fs;

% --- Plot average radial pressure for low/medium/high stiffness ---
figure;
hold on;
colors = {'b','g','r'};
labels = {'Low PWV','Medium PWV','High PWV'};
for k = 1:3
    groupIdx = (bin == k);
    disp(['Group ', labels{k}, ' has ', num2str(sum(groupIdx)), ' subjects']);
    if any(groupIdx)
        meanP = mean(P_mat(groupIdx,:), 1);
        plot(t_P, meanP, 'Color', colors{k}, 'LineWidth', 2);
    end
end
legend(labels); xlabel('Time (s)'); ylabel('Pressure (a.u. or mmHg)');
title('Average Radial Pressure by Stiffness Group');
hold off;


%% --- Part 3: Classical Feature-based Regression ---
RI     = [haemods(plaus_idx).RI]';
SI     = [haemods(plaus_idx).SI]';
AGImod = [haemods(plaus_idx).AGI_mod]';

T = table(RI, SI, AGImod, PWV_cf, 'VariableNames', {'RI','SI','AGImod','PWV_cf'});

mdl = fitlm(T, 'PWV_cf ~ RI + SI + AGImod');
disp(mdl);

% --- ML on Full Waveforms ---
X = PPG_mat;        % Features: waveform samples (subjects x time)
y = PWV_cf;         % Target: stiffness
N = size(X,1);

cv = cvpartition(N, 'HoldOut', 0.2);
idxTrain = training(cv);
idxTest = test(cv);

X_train = X(idxTrain,:);
y_train = y(idxTrain);
X_test = X(idxTest,:);
y_test = y(idxTest);

tree = fitrtree(X_train, y_train);
y_pred = predict(tree, X_test);

fprintf('Correlation (r) between true and predicted PWV: %.2f\n', corr(y_pred, y_test));

figure;
scatter(y_test, y_pred);
hold on;
plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], 'k--', 'LineWidth', 2);
hold off;
xlabel('True PWV'); ylabel('Predicted PWV');
title('Test Set Prediction: Regression Tree');
grid on;

%% --- PART 4: Machine Learning on Full Wrist PPG Waveforms ---

% Input features: Each subject's full wrist PPG waveform
X = PPG_mat;     % size: [N_subjects x N_timepoints]

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


%% Example plausible subject's wrist PPG waveform
t = (0:minLen-1) / fs;          % Time vector in seconds
figure;
plot(t, PPG_mat(1,:), 'b-', 'LineWidth', 2);
xlabel('Time (s)');
ylabel('PPG (a.u.)');
title('Example Wrist PPG Waveform (First Subject)');
grid on;


