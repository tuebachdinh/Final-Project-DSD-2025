function part6_signal_processing(waves, fs)
%PART6_SIGNAL_PROCESSING Signal processing algorithms for feature extraction

addpath('/Users/edintue/Downloads/Final-Project-DSD-2025/algorithms/');

subject_idx = 1;
signal = waves.PPG_Radial(subject_idx, :);

S = struct();
S.v = signal(:);
S.fs = fs;

% Part 6.1: PulseAnalyse10
options = struct();
options.do_plot = true;

[cv_inds, fid_pts, pulses, S_filt] = PulseAnalyse10(S, options);

disp(cv_inds);
disp(fid_pts);
disp(pulses);

dic_idx = fid_pts.dic;
figure; plot(S.v);
hold on; plot(dic_idx, S.v(dic_idx), 'ro', 'MarkerSize', 8, 'DisplayName', 'Dicrotic Notch');
legend('Signal','Dicrotic Notch');
title('PPG with Detected Dicrotic Notch');
xlabel('Sample'); ylabel('Amplitude');

% Part 6.2: Gaussian fitting
[fitCurve, params, p1_idx, p2_idx] = fitTwoGaussiansPPG(S.v);

figure;
plot(S.v, 'b', 'DisplayName', 'Original Signal'); hold on;
plot(fitCurve, 'r--', 'DisplayName', 'Fitted 2-Gaussian');
plot([p1_idx, p2_idx], fitCurve([p1_idx, p2_idx]), 'ko', 'MarkerFaceColor','g','MarkerSize',10);
legend('show');
title('Gaussian Fitting: P1 & P2 Detection');
xlabel('Sample'); ylabel('Amplitude');

% Part 6.3: Frequency and morphology features
features = extractFreqMorphFeatures(S.v, S.fs);
disp('Frequency and Morphology Features:');
disp(features);

% Part 6.4: Signal quality
template = S.v;
[sqi, isGood] = computeSimpleSQI(S.v, template);
fprintf('SQI value: %.2f | Is Good: %d\n', sqi, isGood);

% Part 6.5: New PulseAnalyse
S_new = struct();
S_new.v = signal(:);
S_new.fs = fs;

options_new = struct();
options_new.do_plot = true;
options_new.do_filter = 1;
options_new.do_quality = 1;
options_new.normalise_pw = 1;
options_new.beat_detector = 'IMS';

[pw_inds_new, fid_pts_new, pulses_new, sigs_new] = PulseAnalyse(S_new, options_new);

fprintf('\n=== New PulseAnalyse Results ===\n');
if ~isempty(pw_inds_new)
    fprintf('Augmentation Index (AI): %.2f%%\n', pw_inds_new.AI.v);
    fprintf('Reflection Index (RI): %.2f\n', pw_inds_new.RI.v);
    fprintf('Stiffness Index (SI): %.2f m/s\n', pw_inds_new.SI.v);
    fprintf('Crest Time (CT): %.3f s\n', pw_inds_new.CT.v);
    fprintf('Delta T: %.3f s\n', pw_inds_new.delta_t.v);
end

fprintf('Part 6: Signal processing completed\n');
end