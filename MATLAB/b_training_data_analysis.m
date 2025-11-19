%[text] # Toolkit for training data analysis/improvement
%[text] Gives the user a numbered list of classes, so that the user can choose, which class to analyze
%% Signal QA for "Training Data - labelled"
% Scans class folders, summarizes sample lengths, and plots all samples and outliers.
% Assumptions:
% - Base folder has subfolders named by class.
% - Each subfolder contains *.csv files (lower-case extension).
% - Each CSV is numeric with rows=time and 13 columns=sensors.

baseDir = fullfile(pwd, 'Training Data - labelled');
assert(isfolder(baseDir), 'Folder not found: %s', baseDir);

%% 1) Scan class folders and present numbered list
subdirs = dir(baseDir);
subdirs = subdirs([subdirs.isdir] & ~ismember({subdirs.name}, {'.','..'}));
classNames = {subdirs.name}';

fprintf('Classes found:\n');
for i = 1:numel(classNames)
    fprintf('%2d) %s\n', i, classNames{i});
end
%%
%[text] Full range of classes:
%% 2) User-selectable class range
startClass = 1;
endClass   = numel(classNames);  % change these to analyze a subset
%[text] Uncomment and set these variables to analyze only one class or a subrange of classes:
startClass = 4;
endClass = 17;
%[text] Do some checks:
% Clamp to valid range in case user edits
startClass = max(1, min(startClass, numel(classNames)));
endClass   = max(startClass, min(endClass, numel(classNames)));
%[text] Edit this variable for other permissable deviations:
%% 3) Outlier threshold (fraction of mean length)
thresholdRange = 0.4;  % e.g., 0.3 => ±30% from mean
%[text] Finally, the analysis begins:
%% 4) Analyze each selected class
for c = startClass:endClass
    className = classNames{c};
    folder = fullfile(baseDir, className);

    files = dir(fullfile(folder, '*.csv'));  % Windows: case-insensitive
    if isempty(files)
        fprintf('\n[%s]\nPath: %s\n(No CSV files found.)\n', className, folder);
        continue
    end

    % Read lengths and stash first-channel signals for plotting
    L = zeros(numel(files),1);
    sig1 = cell(numel(files),1);      % first sensor column for quick visualization
    fnames = {files.name}';

    for k = 1:numel(files)
        X = readmatrix(fullfile(files(k).folder, files(k).name));
        L(k) = size(X,1);
        sig1{k} = X(:,1);             % visualize column 1 to keep plots readable, in our case column 1 contains cumulative signal data
    end

    % --- 4.1) Display class name with full path
    fprintf('\n[%s]\nPath: %s\n', className, folder);

    % --- 4.2) Length distribution stats
    meanL = mean(L);
    stdL  = std(L);
    varL  = var(L);
    medL  = median(L);
    minL  = min(L);
    maxL  = max(L);

    fprintf('Samples: %d | mean=%.2f, std=%.2f, var=%.2f, median=%.2f, min=%d, max=%d\n', ...
        numel(L), meanL, stdL, varL, medL, minL, maxL);

    % --- 4.3) Plot where all data samples are shown (overlay first sensor)
    figure('Name', sprintf('%s — All samples (column 1)', className));
    hold on
    for k = 1:numel(sig1)
        plot(sig1{k});  % variable lengths; each line runs to its own length
    end
    hold off
    xlabel('Sample index'); ylabel('Amplitude (column 1)');
    title(sprintf('%s — all samples (n=%d)', className, numel(sig1)));

    % --- 4.4) Identify outliers by length (±thresholdRange of mean)
    lowerThr = (1 - thresholdRange) * meanL;
    upperThr = (1 + thresholdRange) * meanL;
    shortIdx = find(L < lowerThr);
    longIdx  = find(L > upperThr);

    % --- 4.5) Plot and list "30%% shorter" group
    if ~isempty(shortIdx)
        figure('Name', sprintf('%s — Shorter than %.0f%% of mean', className, thresholdRange*100));
        hold on
        for ii = shortIdx(:).'
            plot(sig1{ii}, 'DisplayName', fnames{ii});
        end
        hold off
        xlabel('Sample index'); ylabel('Amplitude (column 1)');
        title(sprintf('%s — length < %.0f%% of mean (n=%d)', className, thresholdRange*100, numel(shortIdx)));
        legend('show', 'Interpreter','none', 'Location','northwest');
        grid on

        fprintf('Shorter than %.0f%% of mean (files):\n', thresholdRange*100);
        fprintf('  %s\n', fnames{shortIdx});
    else
        fprintf('No samples shorter than %.0f%% of mean.\n', thresholdRange*100);
    end

    % --- 4.5) Plot and list "30%% longer" group
    if ~isempty(longIdx)
        figure('Name', sprintf('%s — Longer than %.0f%% of mean', className, thresholdRange*100));
        hold on
        for ii = longIdx(:).'
            plot(sig1{ii}, 'DisplayName', fnames{ii});
        end
        hold off
        xlabel('Sample index'); ylabel('Amplitude (column 1)');
        title(sprintf('%s — length > %.0f%% of mean (n=%d)', className, thresholdRange*100, numel(longIdx)));
        legend('show', 'Interpreter','none', 'Location','northwest');
        grid on

        fprintf('Longer than %.0f%% of mean (files):\n', thresholdRange*100);
        fprintf('  %s\n', fnames{longIdx});
    else
        fprintf('No samples longer than %.0f%% of mean.\n', thresholdRange*100);
    end
end
%[text] 

%[appendix]{"version":"1.0"}
%---
%[metadata:view]
%   data: {"layout":"onright","rightPanelPercent":40}
%---
