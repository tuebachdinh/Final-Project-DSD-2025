%% Load all data and search tool functions
load('Fictive_database.mat'); % contains NETWORK, COMPUTED_PHYSIO, etc.
load('RADIAL_physio.mat');    % contains RADIAL_PHYSIO cell array
addpath('/MATLAB Drive/Final Project/MatlabSearchTools');

%% Explore Example Subject and Signal
i = 1; % subject indice
subj = RADIAL_PHYSIO{i};
t = subj.ONE_CYCLE(:,1); % time (s)
P = subj.ONE_CYCLE(:,2); % pressure (Pa)
Q = subj.ONE_CYCLE(:,3); % flow (m^3/s)
A = subj.ONE_CYCLE(:,4); % area (m^2)

figure;
plot(t, P); xlabel('Time (s)'); ylabel('Pressure (Pa)');
title('Radial Pressure Waveform, Example Subject');

%% Feature Extraction for All Subjects
N = numel(RADIAL_PHYSIO);
features = zeros(N, 6);  % [PulseHeight, UpstrokeTime, UpSlope, DownSlope, AUC, SystTime]
labels = zeros(N,1);     % cfPWV (central stiffness, label)

for i = 1:N
    subj = RADIAL_PHYSIO{i};
    P = subj.ONE_CYCLE(:,2);
    t = subj.ONE_CYCLE(:,1);
    
    % 1. Pulse Height
    pmin = min(P);
    pmax = max(P);
    features(i,1) = pmax - pmin;
    
    % 2. Upstroke time (from min to max)
    [~, imin] = min(P);
    [~, imax] = max(P);
    features(i,2) = t(imax) - t(imin);
    
    % 3. Upstroke Slope (max derivative in rising phase)
    dP = diff(P) ./ diff(t);
    if imax > imin
        features(i,3) = max(dP(imin:imax-1));
    else
        features(i,3) = NaN;
    end
    
    % 4. Downstroke Slope (min derivative after peak)
    if imax < length(dP)
        features(i,4) = min(dP(imax:end));
    else
        features(i,4) = NaN;
    end
    
    % 5. Area Under Curve (AUC)
    features(i,5) = trapz(t, P);
    
    % 6. Systolic time (time above mean)
    pmean = mean(P);
    syst_idx = find(P > pmean);
    features(i,6) = (length(syst_idx) / length(P)) * (t(end) - t(1));
    
    % 7. Label: cfPWV (central aortic stiffness)
    labels(i) = COMPUTED_PHYSIO.PWV_PATH{i}.FootToFoot(2); % 2=cfPWV
end

%check git 