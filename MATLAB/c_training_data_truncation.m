%[text] # Truncation of the initial part of the training data 
%[text] A streamlined procedure for going throught the training data and truncating the initial portion of the signal, which frequently contains no useful information.
%[text] First we go through all the files and determine the truncation point, no CSVs are truncated at this point:
%% Interactive truncation curator (MATLAB R2024a+)
% Iterates through CSV files under "Training Data - labelled", shows a plot,
% user enters number of initial samples to truncate, presses Enter to advance.

function truncationCurator % wrapper for variable visibility
    
    % -------- settings --------
    baseDir = fullfile(pwd, 'Training Data - labelled');  % root folder
    channelToPlot = 1;    % which column (1..13) to display
    autosavePath  = fullfile(pwd, 'truncate_choices.csv'); % output summary
    assert(isfolder(baseDir), 'Folder not found: %s', baseDir);
    
    % -------- gather files (recursive) --------
    D = dir(fullfile(baseDir, '**', '*.csv'));   % Windows: case-insensitive
    D = D(~[D.isdir]);
    if isempty(D), error('No CSV files found under %s', baseDir); end
    N = numel(D);
    fullpaths = fullfile({D.folder}', {D.name}');
    filenames = {D.name}';
    
    % -------- storage for decisions --------
    truncateN = NaN(N,1);
    sampleLen = NaN(N,1);
    
    % -------- UI --------
    f = uifigure('Name','Truncation Curator','Position',[100 100 1100 700]);
    gl = uigridlayout(f, [3 1]);
    gl.RowHeight = {40, '1x', 60};
    gl.ColumnWidth = {'1x'};
    
    % top bar
    hdr = uigridlayout(gl, [1 4]); hdr.Layout.Row = 1; hdr.ColumnWidth = {300, '1x', 220, 220};
    uicontrolDummy = uilabel(hdr); uicontrolDummy.Text = ' ';  %#ok<NASGU>
    lblFile = uilabel(hdr, 'Text','', 'FontWeight','bold'); lblFile.Layout.Column = 2;
    spn = uispinner(hdr, 'Limits',[0 1], 'Step',1, 'Value',0, 'Editable','on'); spn.Layout.Column = 3;
    spn.Tooltip = 'Initial samples to truncate';
    btnNext = uibutton(hdr, 'Text','Next (Enter)', 'ButtonPushedFcn',@(src,evt)acceptAndNext());
    btnNext.Layout.Column = 4;
    
    % plot
    ax = uiaxes(gl); ax.Layout.Row = 2; grid(ax,'on');
    ax.XLabel.String = 'Sample index'; ax.YLabel.String = sprintf('Channel %d', channelToPlot);
    ttl = title(ax, '');  %#ok<NASGU>
    ax.ButtonDownFcn = @(src,evt) clickToSpinner(src); % clicks insert the X coordinate to the spinner
    
    % footer
    foot = uigridlayout(gl, [1 3]); foot.Layout.Row = 3; foot.ColumnWidth = {'1x', 140, 140};
    lblHint = uilabel(foot, 'Text','Enter truncation, press Enter. Shortcuts: ← Backspace = previous; Q = quit; S = save.');
    btnPrev = uibutton(foot, 'Text','Previous', 'ButtonPushedFcn',@(s,e)goPrev());
    btnSave = uibutton(foot, 'Text','Save CSV', 'ButtonPushedFcn',@(s,e)saveNow());
    
    % live vertical marker
    vline = []; sigLine = [];
        
    % keyboard shortcuts
    f.WindowKeyPressFcn = @(src,evt)keyHandler(evt);
    
    % index pointer
    i = 1;
    
    % initial load
    loadAndShow(i);
    
    % ------- callbacks & helpers (nested functions capture workspace) -------
    function keyHandler(evt)
        switch lower(evt.Key)
            case {'return','enter'}
                acceptAndNext();
            case {'backspace','leftarrow'}
                goPrev();
            case 'q'
                selection = uiconfirm(f,'Quit and export CSV?','Quit','Options',{'Yes','No'},'DefaultOption',2);
                if strcmp(selection,'Yes'), saveNow(); close(f); end
            case 's'
                saveNow();
        end
    end
    
    function loadAndShow(idx)
        % read current sample (lazy load)
        X = readmatrix(fullpaths{idx});
        if size(X,2) < channelToPlot
            error('File %s has only %d columns; channelToPlot=%d.', filenames{idx}, size(X,2), channelToPlot);
        end
        y = X(:, channelToPlot);
        L = numel(y);
        sampleLen(idx) = L;
    
        % prepare spinner limits/value
        spn.Limits = [0 max(0,L)];    % allow 0..L
        if isnan(truncateN(idx)), spn.Value = 0; else, spn.Value = truncateN(idx); end
    
        % plot/update (reuse graphics handles for speed)
%        if isempty(sigLine) || ~isvalid(sigLine)   % unclickable X axes
%            sigLine = plot(ax, 1:L, y); hold(ax,'on');
%            vline = xline(ax, spn.Value, '--');
%            hold(ax,'off');
%        else
%            set(sigLine, 'XData', 1:L, 'YData', y);
%            if ~isvalid(vline), vline = xline(ax, spn.Value, '--'); end
%            vline.Value = spn.Value;
%        end

        % draw/update plot & marker (ensure clicks reach axes)
        if isempty(sigLine) || ~isvalid(sigLine)
            cla(ax); hold(ax,'on');
            sigLine = plot(ax, 1:L, y, 'PickableParts','none');   % let clicks reach axes
            vline   = xline(ax, spn.Value, '--', 'HitTest','off');% marker ignores clicks
            hold(ax,'off');
        else
            set(sigLine, 'XData', 1:L, 'YData', y, 'PickableParts','none');
            if isempty(vline) || ~isvalid(vline)
                vline = xline(ax, spn.Value, '--', 'HitTest','off');
            else
                vline.Value = spn.Value;
                try, vline.HitTest = 'off'; end
            end
        end


        ax.XLim = [1, max(2,L)];   % avoid degenerate axes
        ax.YLimMode = 'auto';
        ax.Title.String = sprintf('%d/%d   (Length = %d)', idx, N, L);
        lblFile.Text = sprintf('%s', filenames{idx});
    
        % update on spinner change (live move of xline)
        spn.ValueChangingFcn = @(s,e) moveMarker(e.Value);
        spn.ValueChangedFcn  = @(s,e) moveMarker(e.Value);
    end
    
    function moveMarker(val)
        if isvalid(vline), vline.Value = val; end
    end
    
    function acceptAndNext()
        truncateN(i) = round(spn.Value);
        % advance
        if i < N
            i = i + 1;
            loadAndShow(i);
        else
            uialert(f,'Reached last sample. Exporting CSV.','Done');
            saveNow();
        end
    end
    
    function goPrev()
        if i > 1
            truncateN(i) = round(spn.Value);
            i = i - 1;
            loadAndShow(i);
        end
    end
    
    function saveNow()
        T = table(filenames, fullpaths, sampleLen, truncateN, ...
            'VariableNames', {'FileName','FullPath','Length','TruncateN'});
        try
            writetable(T, autosavePath);
            uialert(f, sprintf('Saved to:\n%s', autosavePath), 'Saved');
        catch ME
            uialert(f, sprintf('Failed to save:\n%s', ME.message), 'Save error', 'Icon','warning');
        end
    end

    function clickToSpinner(ax)
        cp = ax.CurrentPoint;         % 2×3; use first row, x column
        x  = round(cp(1,1));          % integer samples
        % clamp to spinner limits
        x  = max(spn.Limits(1), min(spn.Limits(2), x));
        spn.Value = x;                % updates spinner
        moveMarker(x);                % moves the vertical xline
    end


end % wrapper end


truncationCurator; % call the wrapper
%%
%[text] Now we are ready to actually truncate the files:
%% OVERWRITE originals after backing up (robust to header renaming)
baseDir    = fullfile(pwd, 'Training Data - labelled');
choicesPath = fullfile(pwd, 'truncate_choices.csv');
backupRoot  = fullfile(pwd, "Training Data - backup " + datestr(now,'yyyy-mm-dd_HHMMSS'));

assert(isfolder(baseDir), 'Base folder not found: %s', baseDir);
assert(isfile(choicesPath), 'Choices CSV not found: %s', choicesPath);

% Force comma delimiter + preserve headers
opts = detectImportOptions(choicesPath, 'Delimiter', ',', ...
    'ReadVariableNames', true, 'VariableNamingRule','preserve', 'TextType','string');
%opts.DecimalSeparator   = '.';
%opts.ThousandsSeparator = '';
T = readtable(choicesPath, opts);

% Header cleanup + canonical names
vn = string(T.Properties.VariableNames);
vn = erase(vn, char(65279));
vn = strtrim(vn);
T.Properties.VariableNames = cellstr(vn);
expected = ["FileName","FullPath","Length","TruncateN"];
norm = @(s) lower(regexprep(string(s),'[^a-z0-9]',''));
for e = expected
    j = find(norm(T.Properties.VariableNames) == norm(e), 1);
    if ~isempty(j), T.Properties.VariableNames{j} = char(e); end
end

% Sanity check
assert(all(ismember(["FullPath","TruncateN"], string(T.Properties.VariableNames))), ...
    'CSV must contain FullPath and TruncateN columns.');

% Make backup root
mkdir(backupRoot);

% Process each file
for i = 1:height(T)
    src = T.FullPath(i);
    N   = double(T.TruncateN(i));
    if ismissing(src)
        warning('Row %d has missing FullPath; skipping.', i); 
        continue
    end
    if isnan(N) || N < 0, N = 0; end

    % Read, truncate
    X = readmatrix(src);
    L = size(X,1);
    if N >= L
        warning('Skipping (truncateN >= length) for "%s" (%d >= %d).', src, N, L);
        continue
    end
    Y = X(N+1:end, :);

    % Compute backup folder path mirroring class subfolders
    [srcFolder, baseName, ~] = fileparts(src);
    relFolder = "";
    if startsWith(srcFolder, baseDir)
        relFolder = extractAfter(srcFolder, strlength(baseDir));
        if startsWith(relFolder, filesep), relFolder = extractAfter(relFolder, 1); end
    end
    bkpFolder = fullfile(backupRoot, relFolder);
    if ~isfolder(bkpFolder), mkdir(bkpFolder); end

    % Backup original, then overwrite original
    copyfile(src, fullfile(bkpFolder, baseName + ".csv"));  % backup
    writematrix(Y, src, 'Delimiter',';');                                    % overwrite
end

disp('Done: originals overwritten; backups saved under:');
disp(backupRoot);


%[appendix]{"version":"1.0"}
%---
%[metadata:view]
%   data: {"layout":"onright","rightPanelPercent":40}
%---
