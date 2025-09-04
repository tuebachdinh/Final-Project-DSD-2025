function model_interpretability(net, waves, PWV_cf, model_type)
%MODEL_INTERPRETABILITY Analyze which parts of input are most important
% Inputs:
%   net        - trained network (CNN / GRU / TCN)
%   waves      - struct with fields .PPG_Radial and/or .A_Radial  [N x T]
%   PWV_cf     - target vector [N x 1]
%   model_type - 'CNN' | 'GRU' | 'TCN' (used for titles / filenames)

addpath('../utils/others');

fprintf('\n=== Model Interpretability Analysis ===\n');

% ---------- Build seqData & y_true from waves/PWV_cf ----------
[Xppg, Xarea, y, nCh] = prepare_inputs(waves, PWV_cf);
seqData = pack_sequences(Xppg, Xarea, nCh);

% ---------- Subset for computational efficiency ----------
n_samples = min(50, numel(seqData));
idx = randperm(numel(seqData), n_samples);
data_subset = seqData(idx);
y_subset    = y(idx); %#ok<NASGU> % (kept for future use if needed)

% ---------- Baseline predictions ----------
y_pred = predict(net, data_subset);

% ---------- 1) Occlusion Analysis ----------
fprintf('Running Occlusion Analysis...\n');
occlusion_importance = occlusion_analysis(net, data_subset, y_pred);

% ---------- 2) Perturbation Analysis (SHAP-like) ----------
fprintf('Running Perturbation Analysis...\n');
perturbation_importance = perturbation_analysis(net, data_subset, y_pred);

% ---------- Visualize + Save ----------
visualize_importance(occlusion_importance, perturbation_importance, model_type, nCh);
save_figure(sprintf('interpretability_%s', lower(model_type)), 10);

save(sprintf('part10_interpretability_%s.mat', lower(model_type)), ...
     'occlusion_importance', 'perturbation_importance');

fprintf('Interpretability finished and saved.\n');
end


% ===================== LOCAL HELPERS =====================

function [Xppg, Xarea, y, nCh] = prepare_inputs(waves, PWV_cf)
hasPPG  = isfield(waves,'PPG_Radial') && ~isempty(waves.PPG_Radial);
hasArea = isfield(waves,'A_Radial')   && ~isempty(waves.A_Radial);
if ~hasPPG && ~hasArea %#ok<*JMPAS>
    error('waves must contain PPG_Radial and/or A_Radial.');
end
y = PWV_cf(:);

if hasPPG && hasArea
    Xppg  = waves.PPG_Radial;
    Xarea = waves.A_Radial;
    good = all(isfinite(Xppg),2) & all(isfinite(Xarea),2) & isfinite(y);
    Xppg  = Xppg(good,:);  Xarea = Xarea(good,:);  y = y(good);
    nCh = 2;
elseif hasPPG
    Xppg  = waves.PPG_Radial;
    good  = all(isfinite(Xppg),2) & isfinite(y);
    Xppg  = Xppg(good,:);  y = y(good);
    Xarea = [];            nCh = 1;
else
    Xarea = waves.A_Radial;
    good  = all(isfinite(Xarea),2) & isfinite(y);
    Xarea = Xarea(good,:); y = y(good);
    Xppg  = [];            nCh = 1;
end
end

function S = pack_sequences(Xppg, Xarea, nCh)
% z-normalize per sequence and pack to cell sequences [nCh x T]
if nCh==2
    if size(Xppg,1) ~= size(Xarea,1)
        error('PPG and Area must have same number of rows (subjects).');
    end
    Xppg  = zseq(Xppg);
    Xarea = zseq(Xarea);
    N = size(Xppg,1);
    S = cell(N,1);
    for i=1:N
        S{i} = [Xppg(i,:); Xarea(i,:)];
    end
else
    X = Xppg;  if isempty(X), X = Xarea; end
    X = zseq(X);
    N = size(X,1);
    S = cell(N,1);
    for i=1:N
        S{i} = X(i,:); % [1 x T]
    end
end
end

function Xz = zseq(X)
if isempty(X), Xz = []; return; end
mu  = mean(X,2);
sd  = std(X,[],2);
sd(sd==0) = eps;
Xz = (X - mu) ./ sd;
end

function importance = occlusion_analysis(net, seqData, y_baseline)
n_samples  = numel(seqData);
T          = size(seqData{1}, 2);
n_channels = size(seqData{1}, 1);

window_size = max(5, round(0.1 * T));
n_windows   = floor(T / window_size);
importance  = zeros(n_channels, n_windows);

for ch = 1:n_channels
    for w = 1:n_windows
        start_idx = (w-1)*window_size + 1;
        end_idx   = min(w*window_size, T);
        occluded_data = seqData;
        for i = 1:n_samples
            occluded_data{i}(ch, start_idx:end_idx) = 0;
        end
        y_occluded = predict(net, occluded_data);
        importance(ch, w) = mean(abs(y_baseline - y_occluded));
    end
end
end

function importance = perturbation_analysis(net, seqData, y_baseline)
n_samples  = numel(seqData);
T          = size(seqData{1}, 2);
n_channels = size(seqData{1}, 1);

noise_levels    = [0.15 0.30];  % relative to per-sequence std
n_perturbations = 10;           % Monte Carlo
importance      = zeros(n_channels, T);

% precompute per-sequence channel std
chan_std = zeros(n_channels, n_samples);
for i=1:n_samples
    for ch=1:n_channels
        chan_std(ch,i) = std(seqData{i}(ch,:));
    end
end

for ch = 1:n_channels
    for t = 1:T
        c = 1; sens = zeros(numel(noise_levels)*n_perturbations,1);
        for nl = noise_levels
            for rep = 1:n_perturbations
                perturbed = seqData;
                for i = 1:n_samples
                    noise = nl * chan_std(ch,i) * randn();
                    perturbed{i}(ch,t) = perturbed{i}(ch,t) + noise;
                end
                y_pert = predict(net, perturbed);
                sens(c) = mean(abs(y_baseline - y_pert));
                c = c + 1;
            end
        end
        importance(ch, t) = mean(sens);
    end
end
end

function visualize_importance(occlusion_imp, perturbation_imp, model_type, nCh)
figure('Position', [100, 100, 1200, 800]);

% Occlusion heatmap
subplot(2,2,1);
imagesc(occlusion_imp); colorbar;
title(sprintf('%s: Occlusion Importance', model_type));
xlabel('Time Windows'); ylabel('Channel');
if nCh==2
    yticks([1 2]); yticklabels({'PPG','Area'});
else
    yticks(1);     yticklabels({'Ch1'});
end

% Occlusion average over channels
subplot(2,2,2);
plot(mean(occlusion_imp,1), 'LineWidth', 2);
title('Average Occlusion Importance'); grid on;
xlabel('Time Windows'); ylabel('Importance');

% Perturbation heatmap
subplot(2,2,3);
imagesc(perturbation_imp); colorbar;
title(sprintf('%s: Perturbation Sensitivity', model_type));
xlabel('Time Points'); ylabel('Channel');
if nCh==2
    yticks([1 2]); yticklabels({'PPG','Area'});
else
    yticks(1);     yticklabels({'Ch1'});
end

% Perturbation per channel
subplot(2,2,4); hold on;
for ch=1:size(perturbation_imp,1)
    plot(perturbation_imp(ch,:), 'LineWidth', 2, 'DisplayName', sprintf('Ch %d', ch));
end
title('Perturbation Sensitivity by Channel'); grid on;
xlabel('Time Points'); ylabel('Sensitivity'); legend show;

% Console summary
fprintf('\n=== Interpretability Summary ===\n');
[~, max_ch]  = max(mean(occlusion_imp,2));
fprintf('Most important channel (occlusion): %d\n', max_ch);
[~, max_win] = max(mean(occlusion_imp,1));
fprintf('Most important time region (occlusion): Window %d\n', max_win);
[~, max_t]   = max(mean(perturbation_imp,1));
fprintf('Most sensitive time point (perturbation): %d (%.1f%% of cycle)\n', ...
        max_t, 100*max_t/size(perturbation_imp,2));
end
