%[text] Code used to prepare Training data by moving CSV files from the folders organized by test persons to folders organized by the defines classes.
%[text] First define source and target folders
%% Reorganize training CSV files by label
% Run this from the parent folder that contains:
%   ./Training Data
%   ./Training Data - labelled

% --- Setup paths
srcDir = fullfile(pwd, 'Training Data');
dstDir = fullfile(pwd, 'Training Data - labelled');

assert(isfolder(srcDir), 'Source folder not found: %s', srcDir);
if ~isfolder(dstDir), mkdir(dstDir); end


% --- Defined labels (must match exactly, including spaces and digits)
labels = [ ...
    "Head stroke 1","Back stroke 1","Ear scratch 1","Neck scratch 1","Tail scratch 1","Tail tap 1","Tail hit 1","Hold in hands", ...
    "Head stroke 2","Back stroke 2","Ear scratch 2","Neck scratch 2","Tail scratch 2","Tail tap 2","Tail hit 2","Tail pull","Hold strong" ...
];

% Fast exact-match lookup
labelSet = containers.Map(cellstr(labels), num2cell(true(size(labels))));

% --- Collect CSV files recursively (case-insensitive extension)
files = [ dir(fullfile(srcDir, '**', '*.csv')) ];

moved = 0; skipped = 0;

for k = 1:numel(files)
    if files(k).isdir, continue; end
    srcPath = fullfile(files(k).folder, files(k).name);
    [~, baseName, ext] = fileparts(files(k).name);

    % Extract the substring before the first underscore as the candidate label
    tok = regexp(baseName, '^([^_]+)_', 'tokens', 'once');
    if isempty(tok)
        skipped = skipped + 1;
        continue
    end
    label = strtrim(tok{1});

    % Check against the defined label list
    if ~isKey(labelSet, label)
        skipped = skipped + 1;
        continue
    end

    % Ensure destination subfolder exists
    labelFolder = fullfile(dstDir, label);
    if ~isfolder(labelFolder), mkdir(labelFolder); end

    % Build destination path; avoid name collisions by appending " (n)"
    destPath = fullfile(labelFolder, files(k).name);
    if exist(destPath, 'file')
        n = 1;
        while true
            candidate = fullfile(labelFolder, sprintf('%s (%d)%s', baseName, n, ext));
            if ~exist(candidate, 'file')
                destPath = candidate;
                break
            end
            n = n + 1;
        end
    end

    % Move the file
    [ok, msg] = movefile(srcPath, destPath);
    if ok
        moved = moved + 1;
    else
        warning('Failed to move "%s": %s', srcPath, msg);
        skipped = skipped + 1;
    end
end

fprintf('Done. Moved %d file(s) into "%s". Skipped %d file(s).\n', moved, dstDir, skipped);
%%
%[text] 
%[text] Reshuffle the training data set by adding a random sequence after the class name.
%% Rename first numeric token in CSV filenames to a random 7-digit number
% Structure:
%   ./Training Data - labelled/<ClassName>/<ClassName>_NUM1_NUM2.csv
% Action:
%   Replace NUM1 with a random 7-digit number; avoid collisions.

baseDir = fullfile(pwd, 'Training Data - labelled');
assert(isfolder(baseDir), 'Folder not found: %s', baseDir);

rng('shuffle');  % randomize

subdirs = dir(baseDir);
subdirs = subdirs([subdirs.isdir] & ~ismember({subdirs.name}, {'.','..'}));

renamed = 0; skipped = 0;

for i = 1:numel(subdirs)
    className = subdirs(i).name;
    folder = fullfile(baseDir, className);

    files = [dir(fullfile(folder, '*.csv'))];

    for k = 1:numel(files)
        fname = files(k).name;
        % Match: <class>_<num1>_<num2>.csv  (case-insensitive for extension)
        tok = regexp(fname, '^(?<class>[^_]+)_(?<num1>\d+)_(?<num2>\d+)(?<ext>\.csv)$', ...
                     'names', 'once', 'ignorecase');
        if isempty(tok) || ~strcmp(tok.class, className)
            skipped = skipped + 1;
            continue
        end

        oldPath = fullfile(folder, fname);
        [~, ~, ext] = fileparts(fname);  % preserve extension case

        % Generate a unique new 7-digit number (no leading zeros)
        while true
            newNum = randi([1e6, 9999999]);  % 1000000..9999999
            newName = sprintf('%s_%d_%s%s', className, newNum, tok.num2, ext);
            newPath = fullfile(folder, newName);
            if ~exist(newPath, 'file')
                break
            end
        end

        [ok, msg] = movefile(oldPath, newPath);
        if ok
            renamed = renamed + 1;
        else
            warning('Failed to rename "%s": %s', oldPath, msg);
            skipped = skipped + 1;
        end
    end
end

fprintf('Done. Renamed %d file(s); skipped %d file(s).\n', renamed, skipped);


%[appendix]{"version":"1.0"}
%---
%[metadata:view]
%   data: {"layout":"onright","rightPanelPercent":33.9}
%---
