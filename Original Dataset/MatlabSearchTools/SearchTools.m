%=== SearchTools.m
% Examples of the use of search functions within the database of virtual
% subjects.
% Please refer to the Manual.pdf for details on the database and matlab
% structures.

% Marie Willemet
% July 2015, 
% KCL - London, UK

load 'Fictive database.mat';    

%-- Database to analyse: all physiological cases
db = COMPUTED_PHYSIO;
indexes = [1:length(db.INDICES)];


%% Search on the complete database

%-- Search for the case subject with parameters variation (+- x %) specified within db.PARAMETERS(i,:)
% [c_elastic c_muscular D_elastic D_muscular HR SV PVR]
param_values = [-19 0 -10 21 15 0 10];
search_index1 = search_DB_param(db,indexes,param_values);

%-- Search for computed indices within db.INDICES{i}.(var)
% var = 'CO';   interval = [4.1 4.2];
% var = 'ABI';  interval = [1 1.1];
% var = 'PPA';  interval = [1.2 1.3];
var = 'ReflCoef'; interval = [-0.3 -0.2998];
search_index2 = search_DB_indices(db,indexes, var,interval);

%-- Search for pressure value within db.INDICES{i}.Pressures.location(k)
location = 'aortic_root';
% location = 'carotid';
% location = 'brachial';
p_var = 'SBP';
% p_var = 'DBP';
% p_var = 'MBP';
% p_var = 'PP';
interval = [80 80.4];
search_index3 = search_DB_pressures(db,indexes,location,p_var,interval);

%-- Search for cases presenting a PWV value from method ff/th within range
%in db, within db.PWV_PATH{i}.pwv_method(pwv_path)
pwv_method = 'FootToFoot';
% pwv_method = 'Theoretical';
pwv_path = 1; %aPWV
% pwv_path = 2; %cfPWV
% pwv_path = 3; %faPWV
% pwv_path = 4; %baPWV
% pwv_path = 5; %crPWV
% pwv_path = 6; %hfPWV
interval = [4.95 4.96];
search_index4 = search_DB_PWV(db, indexes, pwv_method, pwv_path, interval);


%% Multiple consequent searches (adapt the indexes vector)

%find cases with cardiac output between 4.1 and 4.2 l/min
myCO = search_DB_indices(db,indexes,'CO',[4.1 4.2]);

%within these results, find cases with foot-to-foot aPWV between 4 and 6 m/s
myPWV = search_DB_PWV(db, myCO, 'FootToFoot', 1, [4 6]);

