%[text] # Prepare and load the training data
%[text] Prepare a datastore
sigds = signalDatastore("Training Data - labelled", IncludeSubfolders=true, ReadFcn=@readmatrix);
fname = sigds.Files

% Read one file - testing
%s=read(sigds);
%s=s(:,1:13); % only use first 13 signals

% Load labels
labels = folders2labels("Training Data - labelled");
summary(labels)

% read data set into memory
sigdata = readall(sigds);
sigdata = cellfun(@(tbl) tbl(:,1:13), sigdata, 'UniformOutput', false); % only use first 13 signals


%%
%[text] Combine the last three columns (11-13) with accelerometer vector magnitude. Save it in column 11 and delete columns 12 and 13.
%% Replace accelerometer axes with magnitude
% sigdata: cell array of N×13 numeric matrices (from readall)
% Columns 11,12,13 = accelerometer X,Y,Z

sigdataMagn = cell(size(sigdata));

for k = 1:numel(sigdata)
    X = sigdata{k};  % N×13 array
    % compute vector magnitude of columns 11–13
    mag = sqrt(sum(X(:,11:13).^2, 2));
    % build new matrix: keep cols 1–10, add mag as new col 11
    Y = [X(:,1:10), mag];
    sigdataMagn{k} = Y;  % N×11 array
end

sigdata = sigdataMagn;
%%
%[text] 
%[text] Process loaded data

% Split data for training
idx = splitlabels(labels,[0.7 0.15])
trainidx = idx{1};
validx = idx{2};
testidx = idx{3};

traindata = sigdata(trainidx);
trainlabels = labels(trainidx);

valdata = sigdata(validx);
vallabels = labels(validx);

testdata = sigdata(testidx);
testlabels = labels(testidx);

c = categories(trainlabels);

%%
%[text] Draw bar graph to show sample lengths
sequenceLengths = cellfun(@length,sigdata);
bar(sequenceLengths)
%%
%[text] ## Important - further choose either OPTION 1 or OPTION 2
%[text] 
%[text] **OPTION 1** Padding - uniform length - repeat or truncate signal
%sigpad = padsequences(sigdata,1,Length=400);

%% Repeat/Truncate each signal to padLength (no zero padding)
% Assumptions:
% - padLength is a positive integer
% - sigdata is a cell array; each cell is an N×13 numeric array (time in rows)

padLength = 250;  % set your target length

sigdatapad = cell(size(sigdata));
for k = 1:numel(sigdata)
    x = sigdata{k};
    n = size(x, 1);
    if n >= padLength
        y = x(1:padLength, :);                 % truncate
    else
        reps = ceil(padLength / n);            % repeats needed
        y = repmat(x, reps, 1);                % repeat
        y = y(1:padLength, :);                 % cut to exact length
    end
    sigdatapad{k} = y;                         % N=padLength, 13 columns
end


%%
%[text] Recalculate training/testing/validation set for **OPTION 1**
traindata = sigdatapad(trainidx);
trainlabels = labels(trainidx);

valdata = sigdatapad(validx);
vallabels = labels(validx);

testdata = sigdatapad(testidx);
testlabels = labels(testidx);
%%
%[text] **OPTION 2** Padding - uniform length - repeat or generate multiple training samples from a longer signal, by moving a sliding window along the signal with a defined STEP
%% Parameters (settable)
padLength = 250;   % target number of rows (time steps)
padSize   = 50;    % hop size in rows

%% Outputs
sigdatapad = cell(0,1);   % M×1 cell, each element is padLength×11 double
labelspad  = labels([]);  % M×1 categorical aligned with sigdatapad

%% Build padded/segmented dataset and labels
for i = 1:numel(sigdata)
    X = sigdata{i};      % K×11 double
    [K, D] = size(X);    % D should be 11
    % Assumption per your spec: D == 11; K is variable

    if K == padLength
        % Exactly the right number of rows
        sigdatapad{end+1,1} = X;            % padLength×11
        labelspad(end+1,1)  = labels(i);

    elseif K < padLength
        % Repeat rows until reaching/exceeding padLength, then truncate to exact length
        reps = ceil(padLength / K);
        Xrep = repmat(X, reps, 1);          % (reps*K)×11
        Xpad = Xrep(1:padLength, :);        % padLength×11
        sigdatapad{end+1,1} = Xpad;
        labelspad(end+1,1)  = labels(i);

    else
        % Slide a window over rows with hop = padSize; keep only full windows
        lastStart = K - padLength + 1;
        for s = 1:padSize:lastStart
            Xseg = X(s : s+padLength-1, :); % padLength×11
            sigdatapad{end+1,1} = Xseg;
            labelspad(end+1,1)  = labels(i);
        end
    end
end

%% (Optional) quick sanity peek (commented)
% sizesOK = all(cellfun(@(Y) isequal(size(Y,1), padLength) && size(Y,2)==11, sigdatapad));
% disp([numel(sigdatapad), sizesOK])


%%
%[text] Barchart
sequenceLengths = cellfun(@length,sigdatapad);
bar(sequenceLengths)
%%
%[text] Recalculate index and data for **OPTION 2**
% Split data for training
idx = splitlabels(labelspad,[0.7 0.15])
trainidx = idx{1};
validx = idx{2};
testidx = idx{3};

traindata = sigdatapad(trainidx);
trainlabels = labelspad(trainidx);

valdata = sigdatapad(validx);
vallabels = labelspad(validx);

testdata = sigdatapad(testidx);
testlabels = labelspad(testidx);

c = categories(trainlabels);

%[text] 

%[appendix]{"version":"1.0"}
%---
%[metadata:view]
%   data: {"layout":"onright","rightPanelPercent":30}
%---
