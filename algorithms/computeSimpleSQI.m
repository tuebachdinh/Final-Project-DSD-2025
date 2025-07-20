function [sqi, isGood] = computeSimpleSQI(pulse, template)
% Inputs:
%   pulse    - 1D vector, current beat (normalized)
%   template - 1D vector, average/clean template beat
% Outputs:
%   sqi      - scalar, correlation SQI (0 to 1)
%   isGood   - logical, true if SQI above threshold

pulse = pulse(:);
template = template(:);

% Ensure same length (resample if needed)
if length(pulse) ~= length(template)
    pulse = resample(pulse, length(template), length(pulse));
end

% Compute correlation SQI
sqi = corr(pulse, template);

% Artifact detection: set threshold
thresh = 0.85; % You can adjust based on your data
isGood = (sqi >= thresh);

end
