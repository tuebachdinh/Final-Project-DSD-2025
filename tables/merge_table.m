% Merge, tag, and sort results by R2 (descending)

% --- File names (edit if yours differ)
aug_file  = 'part9_table_augmented_9x4.csv';
clean_file = 'part9_table_clean_9x4.csv';

% --- Read with types inferred
Taug  = readtable(aug_file);
Tclean = readtable(clean_file);

% --- Add Tag column
Taug.Tag   = repmat(string("augmented"), height(Taug), 1);
Tclean.Tag = repmat(string("clean"),     height(Tclean), 1);

% --- (Optional) ensure consistent variable names
% Expected: Model, Input, R2, RMSE, MAE, training_time
% If your headers differ slightly, normalize here:
Taug.Properties.VariableNames   = matlab.lang.makeValidName(Taug.Properties.VariableNames);
Tclean.Properties.VariableNames = matlab.lang.makeValidName(Tclean.Properties.VariableNames);

% --- Merge
Tall = [Taug; Tclean];

% --- Put Tag after Input for readability
if any(strcmp('Tag', Tall.Properties.VariableNames)) && any(strcmp('Input', Tall.Properties.VariableNames))
    Tall = movevars(Tall, 'Tag', 'After', 'Input');
end

% --- Sort by R2 descending (best first)
if any(strcmp('R2', Tall.Properties.VariableNames))
    Tall = sortrows(Tall, 'R2', 'descend');
else
    error('Column "R2" not found. Check your CSV headers.');
end

% --- Save merged table
out_file = 'part9_table_merged_sorted.csv';
writetable(Tall, out_file);
fprintf('Merged & sorted table saved to: %s\n', out_file);

% --- (Optional) show top few rows
disp(Tall(1:min(10,height(Tall)), :));
