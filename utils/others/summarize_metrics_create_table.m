function T = summarize_metrics_create_table(data_tag)
%PART9_MAKE_MODEL_TABLE Build a 9x4 table for GRU/CNN/TCN × (PPG, Area, both)
% Columns: R2, RMSE, MAE, training_time
% Usage:
%   T = part9_make_model_table('augmented');   % or 'clean'

    if nargin < 1 || isempty(data_tag), data_tag = 'augmented'; end

    % --- find repo root / models dir ---
    current_dir = pwd;
    while ~exist(fullfile(current_dir, 'literature-review.md'), 'file')
        parent_dir = fileparts(current_dir);
        if strcmp(current_dir, parent_dir)
            error('Could not find project root (no literature-review.md up the tree).');
        end
        current_dir = parent_dir;
    end
    models_dir = fullfile(current_dir, 'models');

    chans = {'PPG','Area','both'};
    archs = {'GRU','CNN','TCN'};  % 3×3 = 9 rows

    % row template
    Row = struct('Model', "", 'Input', "", 'R2', NaN, 'RMSE', NaN, 'MAE', NaN, 'training_time', NaN);
    rows = repmat(Row, numel(archs)*numel(chans), 1);
    r = 0;

    for a = 1:numel(archs)
        arch = archs{a};
        for c = 1:numel(chans)
            chan_tag = chans{c};
            file = fullfile(models_dir, sprintf('part9_models_%s_%s.mat', data_tag, chan_tag));

            r = r + 1;
            rows(r).Model = string(arch);
            rows(r).Input = string(chan_tag);

            if ~exist(file,'file')
                warning('Missing file: %s', file);
                continue;
            end
            S = load(file);
            if ~isfield(S,'metrics')
                warning('No metrics struct in %s', file);
                continue;
            end
            M = S.metrics;
            if isfield(M, arch)
                rows(r).R2            = getf(M.(arch),'R2');
                rows(r).RMSE          = getf(M.(arch),'RMSE');
                rows(r).MAE           = getf(M.(arch),'MAE');
                rows(r).training_time = getf(M.(arch),'training_time');
            else
                warning('%s metrics missing in %s', arch, file);
            end
        end
    end

    % Make table and order exactly: GRU PPG/Area/both, CNN PPG/Area/both, TCN PPG/Area/both
    T = struct2table(rows);
    [~, ia] = ismember(T.Model, ["GRU","CNN","TCN"]);
    [~, ic] = ismember(T.Input, ["PPG","Area","both"]);
    [~, ord] = sortrows([ia ic],[1 2]);
    T = T(ord, {'Model','Input','R2','RMSE','MAE','training_time'});

    % print + save csv
    fprintf('\n=== Part 9 Metrics Table (%s) ===\n', data_tag);
    disp(T);
    out_csv = fullfile(models_dir, sprintf('part9_table_%s_9x4.csv', data_tag));
    writetable(T, out_csv);
    fprintf('Saved CSV: %s\n', out_csv);
end

function v = getf(S, field)
    if isfield(S, field), v = S.(field); else, v = NaN; end
end
