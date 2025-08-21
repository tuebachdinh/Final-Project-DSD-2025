function [waves, haemods, PWV_cf, age, fs, plaus_idx, data] = part1_data_preparation()
%PART1_DATA_PREPARATION Data Preparation and Wave Extraction

data = load('exported_data/pwdb_data.mat');
data = data.data;

plausible = logical(data.plausibility.plausibility_log);
plausible = plausible(:);
N = sum(plausible);

fs = data.waves.fs;
haemods = data.haemods;
plaus_idx = find(plausible);

age = [haemods(plaus_idx).age]';
PWV_cf = [haemods(plaus_idx).PWV_cf]';
PWV_ba = [haemods(plaus_idx).PWV_ba]';

wave_types = {'P', 'U', 'A', 'PPG'};
sites = {'AorticRoot', 'Radial', 'Brachial', 'Femoral', 'Digital'};

waves = struct();
minLens = struct();

for w = 1:length(wave_types)
    for s = 1:length(sites)
        f = sprintf('%s_%s', wave_types{w}, sites{s});
        if isfield(data.waves, f)
            cellwaves = data.waves.(f)(plausible);
            lens = cellfun(@numel, cellwaves);
            minLens.(f) = min(lens);
            trunc = cellfun(@(x) x(1:minLens.(f)), cellwaves, 'UniformOutput', false);
            waves.(f) = cell2mat(trunc).';
        end
    end
end

A_Radial = waves.A_Radial;
t_A = (0:size(A_Radial,2)-1) / fs;

fprintf('Part 1: Loaded %d plausible subjects\n', N);
end