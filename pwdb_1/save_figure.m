function save_figure(fig_name, part_num)
%SAVE_FIGURE Save figure to images folder if not already exists
%
% Inputs:
%   fig_name - name of the figure (without extension)
%   part_num - part number (1-9)

% Create images directory if it doesn't exist
img_dir = 'images';
if ~exist(img_dir, 'dir')
    mkdir(img_dir);
end

% Create filename with part prefix
filename = sprintf('%s/part%d_%s.png', img_dir, part_num, fig_name);

% Only save if file doesn't exist
if ~exist(filename, 'file')
    saveas(gcf, filename);
    fprintf('Saved: %s\n', filename);
else
    fprintf('Already exists: %s\n', filename);
end

end