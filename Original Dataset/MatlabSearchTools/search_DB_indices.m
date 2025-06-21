function search_index = search_DB_indices(db, indexes, var, interval)
% Function that searches within cases 'indexes' of 
% database 'db' for cases presenting variable 'var' of value within 'interval'.

% db:  database to analyse
% indexes: vector of indexes within db on which search is performed 
% var: variable to look for, specified as string: 'CO', 'ABI', 'PPA', 'ReflCoef'
% interval: vector of lower and upper limits of the variable value: [lower_limit upper_limit]
% search_index: indices of cases satisfying the search

% Marie Willemet
% July 2015, 
% KCL - London, UK

search_index = [];
j =0;
for i=1:length(indexes)
    ix = indexes(i);
    if(db.INDICES{ix}.(var) >= interval(1) && db.INDICES{ix}.(var) <= interval(2))
        j = j+1;
        search_index(j) = ix;
    end
end

