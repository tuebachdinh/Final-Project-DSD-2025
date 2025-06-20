function search_index = search_DB_PWV(db, indexes, pwv_method, pwv_path, interval)
% Function that searches within cases 'indexes' of a database 'db' for cases presenting a
% value of pwv along 'pwv_path' computed with 'pwv_method' within 'range'.

% db:  database to analyse
% indexes: vector of indexes within db on which search is performed 
% pwv_method: algorithm used to compute pwv, specified as string
%        pwv_method = 'FootToFoot', 'Theoretical';
% pwv_path: Path along which the pwv is computed, specified as following integer
%     pwv_path = 1; %aPWV  - aortic root to iliac bifurcation
%              = 2; %cfPWV - carotid to femoral
%              = 3; %faPWV - femoral to ankle
%              = 4; %baPWV - brachial to ankle
%              = 5; %crPWV - carotid to radial 
%              = 6; %hfPWV - heart (aortic root) to femoral
% interval: vector of lower and upper limits of the PWV value: [lower_limit upper_limit]
% search_index: indices of cases satisfying the search

% Marie Willemet
% July 2015, 
% KCL - London, UK

if not(strcmp(pwv_method,'Theoretical'))
    if not(strcmp(pwv_method,'FootToFoot'))
        disp('Verify the PWV method name. Set to FootToFoot by default');
        pwv_method = 'FootToFoot';
    end
end

check = [1:1:6];
if isempty(find(check == pwv_path))
    disp('Verify PWV path specified. Set to carotid-femoral by default.');
    pwv_path = 2;
end

search_index = [];
j =0;
for i=1:length(indexes)
    ix = indexes(i);    
    if(db.PWV_PATH{ix}.(pwv_method)(pwv_path) >= interval(1) && db.PWV_PATH{ix}.(pwv_method)(pwv_path) <= interval(2) )
        j = j+1;
        search_index(j) = ix;
    end
end
