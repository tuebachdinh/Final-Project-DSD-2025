function part2_visualization(waves, PWV_cf, age, fs)
%PART2_VISUALIZATION Plot waveforms by PWV groups and age groups

wave_types = {'P', 'U', 'A', 'PPG'};

% Part 2.1: Plot by PWV groups
plot_by_pwv_groups(waves, PWV_cf, fs, wave_types);

% Part 2.2: Plot by age groups
plot_by_age_groups(waves, age, fs, wave_types);

fprintf('Part 2: Visualization completed\n');
end


function plot_by_pwv_groups(waves, PWV_cf, fs, wave_types)
site = 'Radial';

pwv = PWV_cf(:);
valid = ~isnan(pwv);
q = quantile(pwv(valid), [1/3 2/3]);

bin_PWV = nan(size(pwv));
bin_PWV(pwv <= q(1)) = 1;
bin_PWV(pwv > q(1) & pwv <= q(2)) = 2;
bin_PWV(pwv > q(2)) = 3;

labels_PWV = {
    sprintf('Low PWV (%.1f–%.1f m/s)', min(pwv(valid)), q(1))
    sprintf('Medium PWV (%.1f–%.1f m/s)', q(1), q(2))
    sprintf('High PWV (%.1f–%.1f m/s)', q(2), max(pwv(valid)))
};
colors_PWV = lines(3);

shade_mode = 'sd';
K = 0.5;

figure('Position',[100 100 900 700]);
for i = 1:numel(wave_types)
    subplot(2,2,i); cla; hold on;
    
    f = sprintf('%s_%s', wave_types{i}, site);
    if ~isfield(waves, f) || isempty(waves.(f))
        title(sprintf('%s (%s) missing', wave_types{i}, site));
        axis off; continue;
    end
    
    W = waves.(f);
    t = (0:size(W,2)-1)/fs;
    
    for k = 1:3
        idx = (bin_PWV == k);
        if ~any(idx), continue; end
        
        data_group = W(idx, :);
        n = sum(idx);
        mean_w = mean(data_group, 1, 'omitnan');
        std_w = std(data_group, 0, 1, 'omitnan');
        
        band = K * std_w;
        band_note = sprintf('±%.1fσ band', K);
        
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
    ylabel(get_ylabel(wave_types{i}));
    title(sprintf('%s (%s) by Stiffness Group', wave_types{i}, site));
    grid on; box on; axis tight;
    legend('Location','best');
end
sgtitle(sprintf('Waveforms at %s with %s by PWV Group', site, band_note));
save_figure('waveforms_by_pwv_groups', 2);
end

function plot_by_age_groups(waves, age, fs, wave_types)
site = 'Radial';
unique_ages = [25 35 45 55 65 75];
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
                 'DisplayName', sprintf('%d', unique_ages(k)));
        end
    end
    xlabel('Time (s)');
    ylabel(get_ylabel(wave_types{i}));
    title(sprintf('%s (%s) by Age Group', wave_types{i}, site));
    grid on; box on; axis tight;
    legend('Location','best');
end
sgtitle(sprintf('Waveforms at %s by Age Group', site));
save_figure('waveforms_by_age_groups', 2);
end

function ylabel_str = get_ylabel(wave_type)
switch wave_type
    case 'P',   ylabel_str = 'Pressure (mmHg)';
    case 'U',   ylabel_str = 'Flow velocity (m/s)';
    case 'A',   ylabel_str = 'Luminal area (m^3)';
    case 'PPG', ylabel_str = 'PPG (a.u.)';
    otherwise,  ylabel_str = 'Amplitude';
end
end