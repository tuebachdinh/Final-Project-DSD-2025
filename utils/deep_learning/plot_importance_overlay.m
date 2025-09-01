function plot_importance_overlay(importance_mat_file, waves, sample_idx, model_tag)
% Plot PPG & Area waveforms with sensitivity overlays
%
% importance_mat_file : path to interpretability_augmented_<model>.mat
% waves               : struct with fields PPG_Radial, A_Radial, fs (T=337)
% sample_idx          : subject row to plot (e.g., 2675)
% model_tag           : string for filenames (e.g., 'gru' or 'cnn')

% ---- Load importance results ----
S = load(fullfile('..','models',importance_mat_file));  % expects occlusion_importance [2 x nWindows], perturbation_importance [2 x T]
if ~isfield(S,'occlusion_importance')
    error('The MAT file must contain "occlusion_importance".');
end
occl = S.occlusion_importance;

% Ensure occl has shape [channels x windows]
if size(occl,1) ~= 2 && size(occl,2) == 2
    occl = occl.'; % transpose if it's [nWindows x 2]
elseif size(occl,1) ~= 2 && size(occl,2) ~= 2
    error('occlusion_importance must be size [2 x nWindows] or [nWindows x 2].');
end

% pert may be missing -> use zeros
if isfield(S, 'perturbation_importance') && ~isempty(S.perturbation_importance)
    pert = S.perturbation_importance;  % (channels x T?) or (T x channels?)
    % orient to [2 x T]
    if size(pert,1) ~= 2 && size(pert,2) == 2
        pert = pert.'; 
    elseif size(pert,1) ~= 2 && size(pert,2) ~= 2
        % if it's something unexpected, fall back to zeros later
        pert = [];
    end
else
    pert = [];
end

% ---- Pull one subject waveform ----
ppg  = waves.PPG_Radial(sample_idx, :);
area = waves.A_Radial(sample_idx, :);
fs   = 500;
T    = numel(ppg);
t    = (0:T-1)/fs;

% ---- Upsample occlusion (windowed) to per-sample length, exactly T ----
nWindows = size(occl, 2);
if nWindows < 1
    error('occlusion_importance must have at least 1 window.');
end

% exact repeat counts that sum to T
base = floor(T / nWindows);
rem  = T - base * nWindows;
counts = base * ones(1, nWindows);
% distribute the remaining samples (+1) over the first "rem" windows
if rem > 0
    counts(1:rem) = counts(1:rem) + 1;
end

occl_up = zeros(2, T);
for ch = 1:2
    occl_up(ch, :) = repelem(occl(ch, :), counts);  % exact length T
end

% ---- Prepare perturbation map to length T ----
if isempty(pert)
    pert = zeros(2, T);
else
    % crop or pad (repeat edges) to match T
    if size(pert,2) > T
        pert = pert(:, 1:T);
    elseif size(pert,2) < T
        % pad by repeating last value
        lastcol = pert(:, end);
        pert = [pert, repmat(lastcol, 1, T - size(pert,2))];
    end
end

% ---- Normalize and combine importance ----
occl_n = normalize_safe(occl_up, 2, 'range');  % [0,1] across time per channel
pert_n = normalize_safe(pert,   2, 'range');   % [0,1]
comb   = 0.5*occl_n + 0.5*pert_n;              % simple average; adjust weights if desired

% ---- Plot helper ----
plot_one_channel(t, ppg,  comb(1,:), 'PPG',  'a.u.', sprintf('ppg_importance_%s_%d',  model_tag, sample_idx));
plot_one_channel(t, area, comb(2,:), 'Area', 'm^3',  sprintf('area_importance_%s_%d', model_tag, sample_idx));

end

% ====================== helpers ======================

function Xn = normalize_safe(X, dim, method)
% normalize with guard for constant arrays
if nargin < 3, method = 'range'; end
Xn = zeros(size(X));
switch lower(method)
    case 'range'
        mn = min(X, [], dim);
        mx = max(X, [], dim);
        rng = mx - mn;
        rng(rng==0) = 1; % avoid divide-by-zero for flat signals
        Xn = (X - expand(mn, size(X), dim)) ./ expand(rng, size(X), dim);
    otherwise
        error('Unsupported normalization method.');
end
end

function Y = expand(v, sz, dim)
% expand vector v (size 1 along DIM) to size SZ
rep = ones(1, numel(sz));
rep(dim) = sz(dim);
Y = repmat(v, rep);
end

function plot_one_channel(t, y, importance, labelY, unitY, fname_stub)
% Creates a figure with base waveform and highlighted sensitive segments

% ---- Choose threshold (top 30% most sensitive) ----
thr = quantile(importance, 0.70);
mask = importance >= thr;

% ---- Base plot ----
figure('Position', [120 120 980 360]); hold on;
plot(t, y, 'Color', [0.6 0.6 0.6], 'LineWidth', 1.2); % grey baseline

% ---- Color/width map by importance (optional smooth) ----
segments = mask_to_segments(mask);
for k = 1:size(segments,1)
    i1 = segments(k,1); i2 = segments(k,2);
    seg_idx = i1:i2;

    % map importance to line width in [2.5, 5.5]
    seg_imp = importance(seg_idx);
    lw = 2.5 + 3.0 * normalize_safe(seg_imp, 2, 'range');  % per-segment relative thickness

    % draw segment in small pieces to vary width smoothly
    for j = 1:numel(seg_idx)-1
        plot(t(seg_idx(j):seg_idx(j+1)), y(seg_idx(j):seg_idx(j+1)), ...
            'LineWidth', lw(j), 'Color', [0.10 0.45 0.85]); % blue overlay
    end

    % optional translucent band under the sensitive region
    xx = [t(i1) t(i2) t(i2) t(i1)];
    yy = [min(y) min(y) max(y) max(y)];
    p = patch(xx, yy, [0.10 0.45 0.85], 'FaceAlpha', 0.07, 'EdgeColor', 'none');
    uistack(p, 'bottom');
end

% ---- Importance trace for reference ----
yyaxis right; 
plot(t, rescale(importance, 0, 1), '-', 'LineWidth', 1.2); 
ylabel('Importance (0–1)');
yyaxis left;

xlabel('Time (s)');
ylabel(sprintf('%s (%s)', labelY, unitY));
title(sprintf('%s with Sensitivity Overlay (top 30%% highlighted)', labelY));
grid on; box on;

% ---- Save (uses your save_figure if available) ----
if exist('save_figure','file')
    try
        save_figure(fname_stub, 10);
    catch
        saveas(gcf, [fname_stub '.png']);
    end
else
    saveas(gcf, [fname_stub '.png']);
end
end

function segs = mask_to_segments(mask)
% Return [start, end] indices of contiguous true regions in a logical vector
d = diff([false, mask, false]);
starts = find(d == 1);
ends   = find(d == -1) - 1;
segs   = [starts(:), ends(:)];
end
