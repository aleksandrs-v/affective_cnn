%[text] # Step 4: Prepare and load the training data
%[text] 
%[text] This script loads the truncated training data, applies feature reduction, splits the data into training/validation/test subsets, and converts variable-length recordings into fixed-length windows suitable for CNN training.
%[text] 
%[text] Feature reduction:
%[text] The 13 raw sensor columns are reduced to 11 features. The three accelerometer axes (X, Y, Z) are replaced by their vector magnitude sqrt(X^2 + Y^2 + Z^2), yielding a single orientation-invariant inertial channel. The cumulative capacitive sum (channel 1) is retained alongside the 9 individual capacitive channels.
%[text] 
%[text] Dataset splitting:
%[text] Recordings are split into training (70), validation (15), and test (15) subsets using stratified sampling to preserve the class distribution across all three subsets.
%[text] 
%[text] Windowing (two user-selectable modes):
%[text] 
%[text] OPTION 1 -- Uniform-length padding (loop-based):
%[text] All recordings are padded or truncated to a uniform target length. Recordings shorter than the target are extended by cyclic repetition (looping) rather than zero-padding, preserving the spectral characteristics of periodic gestures. Each recording produces exactly one training sample.
%[text] 
%[text] OPTION 2 -- Sliding-window segmentation (default, recommended):
%[text] Fixed-length windows of 250 samples (2.5 s at 100 Hz) are extracted with a 50-sample hop stride, generating multiple overlapping training instances per recording. This mode implicitly 
%[text] augments the training set and ensures that no collected signal data is discarded, making it especially suitable when participant recruitment is limited.
%[text] 
%[text] Output:
%[text] A labelled dataset (tensor + class labels) saved as a .mat file, ready for direct ingestion by the training stage (Step 5 / Experiment Manager).
%[text] 
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
%[text] Combine the last three columns (11-13) as accelerometer vector magnitude. Save it in column 11 and delete columns 12 and 13.
%[text] Important: replacing (x,y,z) with magnitude simplifies signal processing by CNN, but the directional information is lost.
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
%[text] Here the data is split into Training set (70%), Validation set (15%), and Testing set (15%).
%[text] Testing set is data, which is not used in training, but is used to test the model after training. 

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
%[text] Draw bar graph to show sample lengths.
%[text] Most probably your data will be all of different lengths (unless you made some pre-processing), here is a graph that shows it:
sequenceLengths = cellfun(@length,sigdata);
bar(sequenceLengths)
%%
%[text] ## PADDING
%[text] ### Important - further choose either OPTION 1 or OPTION 2, do not execute both
%[text] 
%[text] **OPTION 1** Padding - uniform length - repeat or truncate signal
%[text] For each data sample: if length \> window, then truncate; if length \< window, then repeat
%[text] Simple, straight-forward approach, long sampels are simply truncated, each data sample results in 1 sample in the training set.
%[text] Note: in our case the length of samples is set to 250 
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
%[text] 
%[text] **OPTION 2** Padding - uniform length - repeat or generate multiple training samples from a longer signal, by moving a sliding window along the signal with a defined STEP while the signal can be sliced into multiple subsets that are added to the training set. Thus, each source sample can generate multiple samples in the training set.
%[text] For each data sample: recursively move the starting point with a predefined step until a full-length sample for the training set can be extracted.
%[text] Longer data samples result in multiple samples in the uniform training set. As the result no information that is contained in longer samples is lost due to truncation and everything is used in the training. With the selection of an appropriate STEP, the resulting model is expected to function more robustly.
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
%[text] Barchart - optional, simply demonstrates that the length of data samples is now uniform.
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
