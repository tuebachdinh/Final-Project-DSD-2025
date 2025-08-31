function part11_summarize_metrics()
%PART11_SUMMARIZE_METRICS Load & print Part 9 metrics from saved .mat files.
% Writes models/part9_summary_metrics.csv

    % ---- locate project root ----
    current_dir = pwd;
    while ~exist(fullfile(current_dir, 'literature-review.md'), 'file')
        parent_dir = fileparts(current_dir);
        if strcmp(current_dir, parent_dir)
            error('Could not find project root (no literature-review.md up the tree).');
        end
        current_dir = parent_dir;
    end
    models_dir = fullfile(current_dir, 'models');

    % ---- expected files ----
    entries = {
        'part9_models_clean_PPG.mat',       'clean',     'PPG';
        'part9_models_clean_Area.mat',      'clean',     'Area';
        'part9_models_clean_both.mat',      'clean',     'both';
        'part9_models_augmented_PPG.mat',   'augmented', 'PPG';
        'part9_models_augmented_Area.mat',  'augmented', 'Area';
        'part9_models_augmented_both.mat',  'augmented', 'both';
    };

    % ---- template struct (ensures identical fields each row) ----
    template = struct( ...
        'Data'      , "", ...
        'Channels'  , "", ...
        'BestModel' , "", ...
        'CNN_R2'    , NaN, ...
        'CNN_MAE'   , NaN, ...
        'CNN_RMSE'  , NaN, ...
        'CNN_Time_s', NaN, ...
        'GRU_R2'    , NaN, ...
        'GRU_MAE'   , NaN, ...
        'GRU_RMSE'  , NaN, ...
        'GRU_Time_s', NaN, ...
        'File'      , ""  ...
    );
    rowList = repmat(template, 0, 1);  % empty struct array with fixed schema

    % ---- read each file safely ----
    for k = 1:size(entries,1)
        fname   = entries{k,1};
        dataTag = entries{k,2};
        chanTag = entries{k,3};
        fpath   = fullfile(models_dir, fname);

        if ~exist(fpath, 'file')
            warning('Missing file: %s', fpath);
            continue;
        end

        S = load(fpath);

        % start with template to avoid dissimilar-structure error
        row = template;
        row.Data      = string(dataTag);
        row.Channels  = string(chanTag);
        row.File      = string(fname);

        % optional fields
        if isfield(S, 'best_model'), row.BestModel = string(S.best_model); end
        if isfield(S, 'metrics')
            m = S.metrics;
            row.CNN_R2     = get_metric(m,'CNN','R2');
            row.CNN_MAE    = get_metric(m,'CNN','MAE');
            row.CNN_RMSE   = get_metric(m,'CNN','RMSE');
            row.CNN_Time_s = get_metric(m,'CNN','training_time');
            row.GRU_R2     = get_metric(m,'GRU','R2');
            row.GRU_MAE    = get_metric(m,'GRU','MAE');
            row.GRU_RMSE   = get_metric(m,'GRU','RMSE');
            row.GRU_Time_s = get_metric(m,'GRU','training_time');
        else
            warning('No metrics in %s', fname);
        end

        rowList(end+1) = row; %#ok<AGROW>
    end

    if isempty(rowList)
        error('No metrics found in %s', models_dir);
    end

    % ---- table, sort, print, save ----
    T = struct2table(rowList);

    % nice ordering
    [~,ixData] = ismember(T.Data, ["clean","augmented"]);
    [~,ixChan] = ismember(T.Channels, ["Area","PPG","both"]);
    [~,ord]    = sortrows([ixData ixChan],[1 2]);
    T = T(ord,:);

    fprintf('\n==== Part 9 Summary ====\n');
    disp(T);

    out_csv = fullfile(models_dir, 'part9_summary_metrics.csv');
    writetable(T, out_csv);
    fprintf('Saved CSV: %s\n', out_csv);
end

% ---- helpers ----
function val = get_metric(m, modelName, fieldName)
    val = NaN;
    if isstruct(m) && isfield(m, modelName)
        s = m.(modelName);
        if isfield(s, fieldName)
            val = s.(fieldName);
        end
    end
end
