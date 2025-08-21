function feature_interpretability(model, X_features, y_true, feature_names)
%FEATURE_INTERPRETABILITY Analyze feature importance for traditional ML models
%
% Inputs:
%   model - trained model (e.g., from fitlm, TreeBagger, etc.)
%   X_features - feature matrix [N x F]
%   y_true - true PWV values
%   feature_names - cell array of feature names

fprintf('\n=== Feature Interpretability Analysis ===\n');

% Get baseline predictions
if isa(model, 'LinearModel')
    y_pred = predict(model, X_features);
    model_type = 'Linear';
elseif isa(model, 'TreeBagger') || isa(model, 'CompactTreeBagger')
    y_pred = predict(model, X_features);
    y_pred = str2double(y_pred);
    model_type = 'Tree';
else
    % Generic prediction
    y_pred = predict(model, X_features);
    model_type = 'Generic';
end

n_samples = size(X_features, 1);
n_features = size(X_features, 2);

% 1. Permutation Importance
fprintf('Computing Permutation Importance...\n');
perm_importance = permutation_importance(model, X_features, y_pred);

% 2. Feature Occlusion (set to mean)
fprintf('Computing Feature Occlusion Importance...\n');
occlusion_importance = feature_occlusion(model, X_features, y_pred);

% Visualize results
visualize_feature_importance(perm_importance, occlusion_importance, ...
                           feature_names, model_type);

% Save results
save(sprintf('feature_interpretability_%s.mat', lower(model_type)), ...
     'perm_importance', 'occlusion_importance', 'feature_names');

end

function importance = permutation_importance(model, X_features, y_baseline)
%PERMUTATION_IMPORTANCE Shuffle each feature and measure prediction change

n_features = size(X_features, 2);
importance = zeros(n_features, 1);

for f = 1:n_features
    % Create permuted data
    X_perm = X_features;
    X_perm(:, f) = X_perm(randperm(size(X_perm,1)), f);  % Shuffle feature f
    
    % Get predictions with permuted feature
    if isa(model, 'LinearModel')
        y_perm = predict(model, X_perm);
    elseif isa(model, 'TreeBagger') || isa(model, 'CompactTreeBagger')
        y_perm = predict(model, X_perm);
        y_perm = str2double(y_perm);
    else
        y_perm = predict(model, X_perm);
    end
    
    % Importance = increase in prediction error
    importance(f) = mean(abs(y_baseline - y_perm));
end

end

function importance = feature_occlusion(model, X_features, y_baseline)
%FEATURE_OCCLUSION Set each feature to its mean and measure impact

n_features = size(X_features, 2);
importance = zeros(n_features, 1);

for f = 1:n_features
    % Create occluded data (set feature to mean)
    X_occluded = X_features;
    X_occluded(:, f) = mean(X_features(:, f));
    
    % Get predictions with occluded feature
    if isa(model, 'LinearModel')
        y_occluded = predict(model, X_occluded);
    elseif isa(model, 'TreeBagger') || isa(model, 'CompactTreeBagger')
        y_occluded = predict(model, X_occluded);
        y_occluded = str2double(y_occluded);
    else
        y_occluded = predict(model, X_occluded);
    end
    
    % Importance = change in prediction
    importance(f) = mean(abs(y_baseline - y_occluded));
end

end

function visualize_feature_importance(perm_imp, occl_imp, feature_names, model_type)
%VISUALIZE_FEATURE_IMPORTANCE Plot feature importance rankings

figure('Position', [100, 100, 1400, 600]);

% Sort features by importance
[perm_sorted, perm_idx] = sort(perm_imp, 'descend');
[occl_sorted, occl_idx] = sort(occl_imp, 'descend');

% Top 15 features for visualization
n_show = min(15, length(feature_names));

subplot(1,2,1);
barh(1:n_show, perm_sorted(1:n_show));
set(gca, 'YTick', 1:n_show, 'YTickLabel', feature_names(perm_idx(1:n_show)));
xlabel('Permutation Importance');
title(sprintf('%s: Permutation Importance', model_type));
grid on;

subplot(1,2,2);
barh(1:n_show, occl_sorted(1:n_show));
set(gca, 'YTick', 1:n_show, 'YTickLabel', feature_names(occl_idx(1:n_show)));
xlabel('Occlusion Importance');
title(sprintf('%s: Occlusion Importance', model_type));
grid on;

% Print top features
fprintf('\n=== Top 5 Most Important Features ===\n');
fprintf('Permutation Importance:\n');
for i = 1:min(5, length(feature_names))
    fprintf('  %d. %s (%.4f)\n', i, feature_names{perm_idx(i)}, perm_sorted(i));
end

fprintf('\nOcclusion Importance:\n');
for i = 1:min(5, length(feature_names))
    fprintf('  %d. %s (%.4f)\n', i, feature_names{occl_idx(i)}, occl_sorted(i));
end

end