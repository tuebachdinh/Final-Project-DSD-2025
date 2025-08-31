function make_stacked_plots(csv_clean, csv_aug)
%PART9_PLOT_GROUPED_WITH_VALUES Grouped bar plots with numeric labels on top.
% Inputs: paths to the 9x4 CSVs produced earlier (clean & augmented).
% Saves 8 PNGs into <repo_root>/images.

    if nargin < 2, error('Usage: part9_plot_grouped_with_values(cleanCSV, augCSV)'); end

    [models_dir, images_dir] = get_out_dirs(); %#ok<ASGLU>
    if ~exist(images_dir,'dir'), mkdir(images_dir); end

    Tclean = readtable(csv_clean);
    Taug   = readtable(csv_aug);

    metrics = {'R2','RMSE','MAE','training_time'};
    for dset = 1:2
        if dset == 1
            T = Tclean; tag = 'Clean';
        else
            T = Taug;   tag = 'Augmented';
        end

        for m = 1:numel(metrics)
            metric = metrics{m};
            [M, modelOrder, inputOrder] = build_matrix(T, metric);

            fh = figure('Position',[120 120 900 520]);
            hb = bar(categorical(modelOrder), M, 'grouped');
            xlabel('Model');
            ylabel(metric_label(metric));
            title(sprintf('%s — %s', tag, metric_label(metric)));
            legend(inputOrder,'Location','bestoutside');
            grid on; box on;

            % Add numeric labels on top of each bar
            add_bar_labels(hb, M, metric);

            % Save to images/
            out_png = fullfile(images_dir, sprintf('part9_%s_%s_grouped.png', lower(tag), lower(metric)));
            try
                exportgraphics(fh, out_png, 'Resolution', 300);
            catch
                saveas(fh, out_png);
            end
            close(fh);
        end
    end

    fprintf('Saved plots to images/ folder.\n');
end

% ----------------- helpers -----------------
function [models_dir, images_dir] = get_out_dirs()
    % Find repo root via presence of literature-review.md
    current_dir = pwd;
    while ~exist(fullfile(current_dir, 'literature-review.md'), 'file')
        parent_dir = fileparts(current_dir);
        if strcmp(current_dir, parent_dir)
            error('Could not find project root (no literature-review.md up the tree).');
        end
        current_dir = parent_dir;
    end
    models_dir = fullfile(current_dir, 'models');
    images_dir = fullfile(current_dir, 'images');
end

function [M, modelOrder, inputOrder] = build_matrix(T, metric)
    modelOrder = {'CNN','GRU','TCN'};
    inputOrder = {'PPG','Area','both'};
    M = NaN(numel(modelOrder), numel(inputOrder));
    for i=1:numel(modelOrder)
        for j=1:numel(inputOrder)
            mask = strcmp(T.Model, modelOrder{i}) & strcmp(T.Input, inputOrder{j});
            if any(mask)
                v = T.(metric)(mask);
                M(i,j) = v(1);
            end
        end
    end
end

function s = metric_label(metric)
    switch lower(metric)
        case 'r2',            s = 'R^2';
        case 'rmse',          s = 'RMSE (m/s)';
        case 'mae',           s = 'MAE (m/s)';
        case 'training_time', s = 'Training Time (s)';
        otherwise,            s = metric;
    end
end

function add_bar_labels(hb, M, metric)
% Puts numeric labels above each bar in a grouped bar chart.
% Works for MATLAB R2019b+ (uses XEndPoints/YEndPoints).
    % Autoscale top with a bit of headroom for text
    maxY = max(M(:), [], 'omitnan');
    if isempty(maxY) || isnan(maxY), maxY = 1; end
    ylim([0, maxY*1.15]);

    % Formatting per metric
    switch lower(metric)
        case 'r2'
            fmt = @(x) sprintf('%.3f', x);
        case {'rmse','mae'}
            fmt = @(x) sprintf('%.3f', x);
        case 'training_time'
            fmt = @(x) sprintf('%.0f', x);  % seconds, integer
        otherwise
            fmt = @(x) sprintf('%.3f', x);
    end

    % Loop over each series (PPG/Area/both)
    for s = 1:numel(hb)
        % Guard: XEndPoints introduced in newer MATLAB. If unavailable, skip labels.
        if ~isprop(hb(s), 'XEndPoints') || ~isprop(hb(s), 'YEndPoints')
            warning('This MATLAB version lacks XEndPoints/YEndPoints; skipping labels.');
            return;
        end
        x = hb(s).XEndPoints;
        y = hb(s).YEndPoints;

        for k = 1:numel(x)
            val = y(k);  % equals the plotted M(k,s)
            if isnan(val), continue; end
            text(x(k), y(k) + 0.02*maxY, fmt(val), ...
                 'HorizontalAlignment','center', 'VerticalAlignment','bottom', ...
                 'FontSize',10);
        end
    end
end
