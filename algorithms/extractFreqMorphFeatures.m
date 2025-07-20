function features = extractFreqMorphFeatures(pulse, fs)
% Inputs:
%   pulse - 1D vector, a single normalized beat (PPG, etc.)
%   fs    - sampling frequency in Hz
% Outputs:
%   features - structure containing frequency and morphology features

pulse = pulse(:);
N = length(pulse);

% Frequency features
Y = fft(pulse);
f = (0:N-1)*(fs/N);
mag = abs(Y)/N;

% Energy in main frequency bands (e.g., <2 Hz, 2-5 Hz, >5 Hz)
band1 = f < 2;
band2 = (f >= 2) & (f < 5);
band3 = f >= 5;

features.energy_band1 = sum(mag(band1).^2);
features.energy_band2 = sum(mag(band2).^2);
features.energy_band3 = sum(mag(band3).^2);
features.total_energy = sum(mag.^2);

% Harmonic ratios (if N is sufficient)
[~, domFreqIdx] = max(mag(2:round(N/2))); % skip DC, find dominant
domFreqIdx = domFreqIdx + 1;
features.dominant_freq = f(domFreqIdx);

% Harmonic ratio example: 2nd/1st
features.harm2over1 = mag(2*domFreqIdx)/mag(domFreqIdx);

% Morphology features
features.skewness = skewness(pulse);
features.kurtosis = kurtosis(pulse);
features.mean = mean(pulse);
features.std = std(pulse);
features.area = trapz((1:N)/fs, pulse); % area under curve

% Time-to-peak, rise/decay times
[~, peakIdx] = max(pulse);
features.time_to_peak = peakIdx / fs;
features.rise_time = (peakIdx - find(pulse(1:peakIdx) < 0.2*max(pulse), 1, 'last')) / fs;
features.decay_time = (find(pulse(peakIdx:end) < 0.2*max(pulse), 1, 'first')) / fs;

end