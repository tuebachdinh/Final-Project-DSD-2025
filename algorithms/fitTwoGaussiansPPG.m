function [fitCurve, params, p1_loc, p2_loc] = fitTwoGaussiansPPG(signal)
    % signal: 1D vector (one normalized beat)
    % Outputs:
    %   fitCurve - fitted curve values
    %   params   - Gaussian parameters [A1, mu1, sigma1, A2, mu2, sigma2]
    %   p1_loc   - estimated sample index of first peak (P1)
    %   p2_loc   - estimated sample index of second peak (P2)
    
    n = length(signal);
    x = (1:n)';
    signal = signal(:);
    
    % Initial guesses (very rough)
    [pks, locs] = findpeaks(signal);
    if length(locs) < 2
        locs = [round(n/3); round(2*n/3)];
        pks = [max(signal(1:locs(1))); max(signal(locs(1):end))];
    end
    mu1 = locs(1); mu2 = locs(end);
    A1 = pks(1); A2 = pks(end);
    sigma1 = n/8; sigma2 = n/8;
    
    % Gaussian model: y = A1*exp(-(x-mu1)^2/(2*sigma1^2)) + A2*exp(-(x-mu2)^2/(2*sigma2^2))
    gauss2 = @(p, x) p(1)*exp(-((x-p(2)).^2)/(2*p(3)^2)) + ...
                     p(4)*exp(-((x-p(5)).^2)/(2*p(6)^2));
    init_params = [A1 mu1 sigma1 A2 mu2 sigma2];
    
    % Fit using nonlinear least squares
    opts = optimset('Display','off');
    params = lsqcurvefit(gauss2, init_params, x, signal, [], [], opts);
    fitCurve = gauss2(params, x);
    
    % Find locations of peaks in the fitted curve
    [~, p1_loc] = max(fitCurve(1:round((params(2)+params(5))/2)));
    [~, p2_loc_rel] = max(fitCurve(round((params(2)+params(5))/2):end));
    p2_loc = p2_loc_rel + round((params(2)+params(5))/2) - 1;
    
    % Optional: plot
    figure;
    plot(x, signal, 'b', 'LineWidth',1.5); hold on;
    plot(x, fitCurve, 'r--', 'LineWidth',2);
    plot([p1_loc, p2_loc], fitCurve([p1_loc, p2_loc]), 'ko','MarkerFaceColor','g','MarkerSize',10);
    legend('Signal','Fitted 2-Gaussian','Peaks (P1,P2)');
    xlabel('Sample'); ylabel('Amplitude');
    title('Two-Gaussian Fitting for PPG Pulse');
    hold off;
end