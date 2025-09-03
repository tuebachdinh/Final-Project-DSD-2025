function [waves_augmented, PWV_cf_augmented] = part7_data_augmentation(waves, PWV_cf, fs)
%PART7_DATA_AUGMENTATION Data augmentation for real-world simulation

addpath('../utils/others');
rng(6);

X_ppg_clean = waves.PPG_Radial;
X_area_clean = waves.A_Radial;
y_clean = PWV_cf(:);

good_aug = all(isfinite(X_ppg_clean),2) & all(isfinite(X_area_clean),2) & isfinite(y_clean);
X_ppg_clean = X_ppg_clean(good_aug,:); 
X_area_clean = X_area_clean(good_aug,:);
y_clean = y_clean(good_aug);

[N_aug, T_aug] = size(X_ppg_clean);
fprintf('Clean data: %d subjects, %d time points\n', N_aug, T_aug);

fprintf('\n=== Applying Real-World Augmentations ===\n');

% PPG Augmentations
% Computes per-subject signal power and injects Gaussian noise for SNR≈25 dB
X_ppg_aug = X_ppg_clean;
sig_power = mean(X_ppg_aug.^2, 2);
noise_power = sig_power ./ (10.^(25/10));
noise = sqrt(noise_power) .* randn(size(X_ppg_aug));
X_ppg_aug = X_ppg_aug + noise;

% Baseline drift (5% amplitude)
t_drift = linspace(0, 1, T_aug);
for i = 1:N_aug
    drift_freq = 0.1 + 0.3*rand(); %random frequency between 0.1Hz and 0.4Hz
    drift = 0.05 * sin(2*pi*drift_freq*t_drift + 2*pi*rand());
    X_ppg_aug(i,:) = X_ppg_aug(i,:) + drift;
end

% Motion artifacts (30% probability) (add a short spike segment)
for i = 1:N_aug
    if rand() < 0.3
        spike_start = randi([1, T_aug-20]);
        spike_dur = randi([5, 15]);
        spike_end = min(spike_start + spike_dur, T_aug);
        spike_t = 1:spike_dur;
        spike = 0.2 * exp(-spike_t/3) .* (2*rand()-1);
        X_ppg_aug(i, spike_start:spike_end-1) = X_ppg_aug(i, spike_start:spike_end-1) + spike(1:length(spike_start:spike_end-1));
    end
end

% Area Augmentations
X_area_aug = X_area_clean;
% Add proportional Gaussian noise (25 dB SNR, relative to signal range)
for i = 1:N_aug 
    sig_range = max(X_area_aug(i,:)) - min(X_area_aug(i,:));
    noise_std = sig_range / (10^(25/20));
    noise = noise_std * randn(1, T_aug);
    X_area_aug(i,:) = X_area_aug(i,:) + noise;
end

% Add baseline drift (2% of signal range)
for i = 1:N_aug
    sig_range = max(X_area_clean(i,:)) - min(X_area_clean(i,:));
    drift_freq = 0.1 + 0.3*rand();
    drift = 0.02 * sig_range * sin(2*pi*drift_freq*t_drift + 2*pi*rand());
    X_area_aug(i,:) = X_area_aug(i,:) + drift;
end

% Add motion artifacts (30% probability, scaled to signal)
for i = 1:N_aug
    if rand() < 0.3
        sig_range = max(X_area_clean(i,:)) - min(X_area_clean(i,:));
        spike_start = randi([1, T_aug-20]);
        spike_dur = randi([5, 15]);
        spike_end = min(spike_start + spike_dur, T_aug);
        spike_t = 1:spike_dur;
        spike = 0.05 * sig_range * exp(-spike_t/3) .* (2*rand()-1);
        X_area_aug(i, spike_start:spike_end-1) = X_area_aug(i, spike_start:spike_end-1) + spike(1:length(spike_start:spike_end-1));
    end
end

y_aug = y_clean;
waves_augmented = struct();
waves_augmented.PPG_Radial = X_ppg_aug;
waves_augmented.A_Radial = X_area_aug;
PWV_cf_augmented = y_aug;

% Visualization
figure('Position', [100, 100, 1400, 800]);
sample_idx = randperm(N_aug, 6);
t_plot = (1:T_aug)/fs;

for i = 1:6
    subplot(3,4,i);
    plot(t_plot, X_ppg_clean(sample_idx(i),:), 'b-', 'LineWidth', 1.5); hold on;
    plot(t_plot, X_ppg_aug(sample_idx(i),:), 'r--', 'LineWidth', 1);
    title(sprintf('PPG Subject %d', sample_idx(i)));
    xlabel('Time (s)'); ylabel('PPG (a.u.)');
    if i == 1, legend('Clean', 'Augmented', 'Location', 'best'); end
    grid on;
end

for i = 1:6
    subplot(3,4,i+6);
    plot(t_plot, X_area_clean(sample_idx(i),:), 'b-', 'LineWidth', 1.5); hold on;
    plot(t_plot, X_area_aug(sample_idx(i),:), 'r--', 'LineWidth', 1);
    title(sprintf('Area Subject %d', sample_idx(i)));
    xlabel('Time (s)'); ylabel('Area (m²)');
    if i == 1, legend('Clean', 'Augmented', 'Location', 'best'); end
    grid on;
end

sgtitle('Data Augmentation: Clean vs Augmented Signals');
save_figure('data_augmentation_comparison', 7);

snr_ppg = 10*log10(mean(var(X_ppg_clean, 0, 2)) / mean(var(X_ppg_aug - X_ppg_clean, 0, 2)));
snr_area = 10*log10(mean(var(X_area_clean, 0, 2)) / mean(var(X_area_aug - X_area_clean, 0, 2)));

fprintf('\n=== Augmentation Quality Metrics ===\n');
fprintf('PPG SNR after augmentation: %.1f dB\n', snr_ppg);
fprintf('Area SNR after augmentation: %.1f dB\n', snr_area);
fprintf('Part 7: Data augmentation completed\n');
end