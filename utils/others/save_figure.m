function save_figure(fig_name, part_num)
%SAVE_FIGURE Save figure to images folder if not already exists
%
% Inputs:
%   fig_name - name of the figure (without extension)
%   part_num - part number (1-9)

% Find project root (contains README.md)
current_dir = pwd;
while ~exist(fullfile(current_dir, 'literature-review.md'), 'file')
    parent_dir = fileparts(current_dir);
    if strcmp(current_dir, parent_dir)
        error('Could not find project root');
    end
    current_dir = parent_dir;
end

% Create images directory at root if it doesn't exist
img_dir = fullfile(current_dir, 'images');
if ~exist(img_dir, 'dir')
    mkdir(img_dir);
end

% Create filename with part prefix
filename = fullfile(img_dir, sprintf('part%d_%s.png', part_num, fig_name));

% Only save if file doesn't exist
if ~exist(filename, 'file')
    saveas(gcf, filename);
    fprintf('Saved: %s\n', filename);
else
    fprintf('Already exists: %s\n', filename);
end

end