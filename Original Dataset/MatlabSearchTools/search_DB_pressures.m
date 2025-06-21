function search_index = search_DB_pressures(db, indexes, location, variable, interval)
% Function that searches within cases 'indexes' of database 'db' for cases presenting
% pressure 'variable' at location 'location' of value within 'interval'.

% db:  database to analyse
% indexes: vector of indexes within db on which search is performed 
% location: artery location, specified as string 
%       location = 'aortic_root', 'carotid', 'brachial'
% variable: pressure variable to look for, specified as string:
%       variable = 'SBP', 'DBP', 'MBP', 'PP'
% interval: vector of lower and upper limits of the variable value: [lower_limit upper_limit]
% search_index: indices of cases satisfying the search

% Marie Willemet
% July 2015, 
% KCL - London, UK

switch variable
    case 'SBP'
        var = 1;
    case 'DBP' 
        var = 2;
    case 'MBP' 
        var = 3;
    case 'PP'
        var = 4;
    otherwise
        var = 1;
        disp('Verify the pressure variable name. Set to SBP by default.')
end

if not(strcmp(location,'aortic_root'))
    if not(strcmp(location,'carotid'))
        if not(strcmp(location,'brachial'))
            disp('Verify the location name. Set to brachial by default');
            location = 'brachial';
        end
    end
end

search_index = [];
j =0;
for i=1:length(indexes)
    ix = indexes(i);
    if(db.INDICES{ix}.Pressures.(location)(var) >= interval(1) && db.INDICES{ix}.Pressures.(location)(var) <= interval(2) )
        j = j+1;
        search_index(j) = ix;
    end
end
