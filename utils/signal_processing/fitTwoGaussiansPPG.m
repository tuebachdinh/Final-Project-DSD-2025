function [fitCurve, params, p1_loc, p2_loc, gof] = fitTwoGaussiansPPG(signal, fs)
    % Improved: Robust fitting with constraints and smoothing
    % Inputs:
    %   signal: 1D vector (one normalized beat)
    %   fs    : (optional) sampling frequency (for diagnostics only)
    % Outputs:
    %   fitCurve - fitted curve values
    %   params   - Gaussian parameters [A1, mu1, sigma1, A2, mu2, sigma2]
    %   p1_loc   - estimated sample index of first peak (P1)
    %   p2_loc   - estimated sample index of second peak (P2)
    %   gof      - structure with goodness-of-fit (RMSE, R^2)
    
    n = length(signal);
    x = (1:n)';
    signal = signal(:);

    % 1. Smoothing (to reduce noise)
    win = max(round(0.01*n), 3); % 1% of the beat, min 3
    smooth_signal = smoothdata(signal, 'gaussian', win);

    % 2. Improved initial guesses using maxima and curvature
    [~, pk1] = max(smooth_signal(1:round(n*0.6)));    % Main systolic peak, first 60%
    [~, pk2] = max(smooth_signal(round(n*0.5):end));  % Shoulder/reflection, last 50%
    pk2 = pk2 + round(n*0.5) - 1;
    if abs(pk2-pk1) < n/6
        % Force P2 far enough from P1
        pk2 = min(n, pk1+round(n/4));
    end
    mu1 = pk1;
    mu2 = pk2;
    A1 = smooth_signal(mu1);
    A2 = smooth_signal(mu2);

    sigma1 = n/10; % narrower
    sigma2 = n/10;
    
    % Parameter lower/upper bounds
    lb = [0,      1,   3,  0,   1,   3];
    ub = [2*A1,  mu2-5, n/2,  2*A2, n, n/2];

    % Model
    gauss2 = @(p, x) p(1)*exp(-((x-p(2)).^2)/(2*p(3)^2)) + ...
                     p(4)*exp(-((x-p(5)).^2)/(2*p(6)^2));
    init_params = [A1, mu1, sigma1, A2, mu2, sigma2];

    % 3. Fit using nonlinear least squares with bounds (lsqcurvefit)
    opts = optimset('Display','off');
    try
        params = lsqcurvefit(gauss2, init_params, x, smooth_signal, lb, ub, opts);
    catch
        warning('Fit failed, using initial guess.');
        params = init_params;
    end
    fitCurve = gauss2(params, x);

    % 4. Find peak locations in the fit (restrict search window)
    mid = round((params(2)+params(5))/2);
    [~, p1_loc] = max(fitCurve(1:mid));
    [~, p2_loc_rel] = max(fitCurve(mid:end));
    p2_loc = p2_loc_rel + mid - 1;

    % 5. Optional: Goodness of fit
    res = signal - fitCurve;
    RMSE = sqrt(mean(res.^2));
    R2 = 1 - sum(res.^2) / sum((signal - mean(signal)).^2);
    gof.RMSE = RMSE;
    gof.R2 = R2;

    % % Optional: plot for diagnostics
    % if nargout==0 || nargin>1
    %     figure;
    %     plot(x, signal, 'b', 'LineWidth',1.5); hold on;
    %     plot(x, fitCurve, 'r--', 'LineWidth',2);
    %     plot([p1_loc, p2_loc], fitCurve([p1_loc, p2_loc]), 'ko','MarkerFaceColor','g','MarkerSize',10);
    %     legend('Signal','Fitted 2-Gaussian','Peaks (P1,P2)');
    %     xlabel('Sample'); ylabel('Amplitude');
    %     title(sprintf('Two-Gaussian Fitting (RMSE=%.3g, R^2=%.2f)', RMSE, R2));
    %     hold off;
    % end
    fprintf('Two-Gaussian Fitting (RMSE=%.3g, R^2=%.2f)', RMSE, R2);
    % Warn if fit is poor
    if R2 < 0.90
        warning('Poor Gaussian fit: R^2=%.2f. Result may be unreliable.', R2);
    end
end
