function count_params(model_file)
%PART9_COUNT_PARAMS Count learnable parameters for CNN / GRU / TCN (Series/DAG/dlnetwork)
%
% Usage:
%   part9_count_params('../models/part9_models_augmented_both_with_TCN.mat')

    if nargin < 1
        error('Provide the path to the saved models .mat file');
    end
    S = load(model_file);

    fprintf('\n=== Parameter Counts in %s ===\n', model_file);

    if isfield(S, 'net_cnn')
        fprintf('CNN: %d parameters\n', count_net_params(S.net_cnn));
    end
    if isfield(S, 'net_gru')
        fprintf('GRU: %d parameters\n', count_net_params(S.net_gru));
    end
    if isfield(S, 'net_tcn')
        fprintf('TCN: %d parameters\n', count_net_params(S.net_tcn));
    end
end

function n = count_net_params(net)
% Handles SeriesNetwork, DAGNetwork, dlnetwork
    n = 0;

    % Case 1: dlnetwork (has Learnables table)
    if isa(net, 'dlnetwork')
        tbl = net.Learnables; % table with Parameters column of dlarrays
        for i = 1:height(tbl)
            val = tbl.Value{i};
            if isnumeric(val) || isdlarray(val)
                n = n + numel(extractdata(val));
            end
        end
        return;
    end

    % Case 2: DAGNetwork or SeriesNetwork
    if isprop(net, 'Layers')
        layers = net.Layers;
        for i = 1:numel(layers)
            L = layers(i);
            % Preferred: LearnableParameters property if present
            if isprop(L, 'LearnableParameters')
                params = L.LearnableParameters;
                for k = 1:numel(params)
                    v = params(k).Value;
                    if ~isempty(v) && (isnumeric(v) || isdlarray(v))
                        n = n + numel(v);
                    end
                end
            else
                % Fallback: common learnable field names on classic layers
                names = {'Weights','Bias','InputWeights','RecurrentWeights', ...
                         'PeepholeWeights','Scale','Offset','Beta','Gamma'};
                for k = 1:numel(names)
                    name = names{k};
                    if isprop(L, name)
                        v = L.(name);
                        if ~isempty(v) && isnumeric(v)
                            n = n + numel(v);
                        end
                    end
                end
            end
        end
        return;
    end

    error('Unsupported network object of class %s', class(net));
end
