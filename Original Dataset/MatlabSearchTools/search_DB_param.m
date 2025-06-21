function search_index = search_DB_param(db,indexes,param_values)
% Search for the case within cases 'indexes' with parameters variation 'param_values' (+- x%) specified within 'db'

% db:  database to analyse
% indexes: vector of indexes within db on which search is performed 
% param_values: vector (1x7) with parameter variations v (in %)
%   param_values = [c_elastic c_muscular D_elastic D_muscular HR SV PVR]
% search_index: index of the subject having the parameters specified

% Marie Willemet
% July 2015, 
% KCL - London, UK

factor = 1 + param_values./100;
simu_param_ix = find(db.PARAMETERS(indexes,1) == factor(1) & db.PARAMETERS(indexes,2) == factor(2) & ...
     db.PARAMETERS(indexes,3) == factor(3) & db.PARAMETERS(indexes,4) == factor(4) & ...
     db.PARAMETERS(indexes,5) == factor(5) & db.PARAMETERS(indexes,6) == factor(6) & ...
     db.PARAMETERS(indexes,7) == factor(7));
search_index = indexes(simu_param_ix);