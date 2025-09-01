function part11_transfer_eval(waves, PWV_cf, which_channels, fs)
% Evaluate a saved Part 9 model (e.g., augmented-trained) on:
%   (A) CLEAN synthetic data
%   (B) Optionally a freshly-augmented copy of the same data (toggle below)
%
% Inputs:
%   model_path      - path to .mat saved by Part 9 (contains best_net, best_model)
%   waves, PWV_cf   - clean synthetic data (struct + vector) as used in Parts 1–9
%   which_channels  - 'ppg' | 'area' | 'both' (must match the saved model)
%   fs              - sampling rate (Hz)
%
% Outputs:
%   Prints R2 / MAE / RMSE for CLEAN (and AUG if enabled) and saves a scatter plot.
    
    if nargin < 4, error('Usage: part9_transfer_eval(model_path, waves, PWV_cf, which_channels, fs)'); end
    assert(isfield(waves,'PPG_Radial') && isfield(waves,'A_Radial'), 'waves must have PPG_Radial and A_Radial');

    % ------------ Load model ------------
    S = load(fullfile('..','models','part9_models_augmented_both.mat'));

    net = S.net_cnn;
    %if isfield(S,'best_model'), fprintf('Loaded best_model: %s\n', string(S.best_model)); end

    % ------------ Build CLEAN seqData ------------
    X_ppg  = waves.PPG_Radial;
    X_area = waves.A_Radial;
    y      = PWV_cf(:);

    good = all(isfinite(X_ppg),2) & all(isfinite(X_area),2) & isfinite(y);
    X_ppg  = X_ppg(good,:); 
    X_area = X_area(good,:);
    y      = y(good);

    % per-sequence z-norm (same as Part 9)
    X_ppg_z  = (X_ppg  - mean(X_ppg,2))  ./ max(std(X_ppg,[],2),  eps);
    X_area_z = (X_area - mean(X_area,2)) ./ max(std(X_area,[],2), eps);

    seq_clean = make_seq(X_ppg_z, X_area_z, which_channels);

    % ------------ Predict & metrics on CLEAN ------------
    yp_clean = predict(net, seq_clean);
    [R2c, MAEc, RMSEc] = metrics_reg(y, yp_clean);
    fprintf('\n== Transfer Eval on CLEAN ==\n');
    fprintf('R^2 = %.4f | MAE = %.4f m/s | RMSE = %.4f m/s\n', R2c, MAEc, RMSEc);


    % ------------ Plot scatter for CLEAN ------------
    figure('Position',[100 100 560 460]);
    scatter(y, yp_clean, 18, 'filled'); grid on; hold on;
    lo = min(y); hi = max(y);
    plot([lo hi],[lo hi],'k--','LineWidth',1.2);
    xlabel('True PWV_{cf} (m/s)'); ylabel('Predicted PWV_{cf} (m/s)');
    title(sprintf('Transfer on CLEAN | R^2=%.3f, MAE=%.3f, RMSE=%.3f', R2c, MAEc, RMSEc));

end

% --------- helpers ---------
function seqData = make_seq(X_ppg, X_area, which_channels)
    N = size(X_ppg,1);  % same N for both
    seqData = cell(N,1);
    switch lower(which_channels)
        case 'ppg'
            for i=1:N, seqData{i} = X_ppg(i,:); end
        case 'area'
            for i=1:N, seqData{i} = X_area(i,:); end
        case 'both'
            for i=1:N, seqData{i} = [X_ppg(i,:); X_area(i,:)]; end
        otherwise
            error('which_channels must be ''ppg'',''area'',''both''.');
    end
end

function [R2, MAE, RMSE] = metrics_reg(y, yhat)
    y = y(:); yhat = yhat(:);
    resid = y - yhat;
    R2    = 1 - sum(resid.^2)/sum((y-mean(y)).^2);
    MAE   = mean(abs(resid));
    RMSE  = sqrt(mean(resid.^2));
end


