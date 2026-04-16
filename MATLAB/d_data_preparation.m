%[text] # Stage 5: Prepare and load the training data
%[text] 
%[text] This script loads the truncated training data, applies feature reduction, splits the data into training/validation/test subsets, and converts variable-length recordings into fixed-length windows suitable for CNN training.
%[text] 
%[text] Feature reduction:
%[text] The 13 raw sensor columns are reduced to 11 features. The three accelerometer axes (X, Y, Z) are replaced by their vector magnitude sqrt(X^2 + Y^2 + Z^2), yielding a single orientation-invariant inertial channel. The cumulative capacitive sum (channel 1) is retained alongside the 9 individual capacitive channels.
%[text] 
%[text] Dataset splitting:
%[text] Recordings are split into training (70%), validation (15%), and test (15%) subsets using stratified sampling to preserve the class distribution across all three subsets.
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
fname = sigds.Files %[output:039d6669]

% Read one file - testing
%s=read(sigds);
%s=s(:,1:13); % only use first 13 signals

% Load labels
labels = folders2labels("Training Data - labelled");
summary(labels) %[output:68846a2b]

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
idx = splitlabels(labels,[0.7 0.15]) %[output:5fc675f4]
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
%[text] At this point, recordings are typically of variable length. The following bar chart visualizes the length distribution:
sequenceLengths = cellfun(@length,sigdata);
bar(sequenceLengths) %[output:90cd383f]
%%
%[text] ## PADDING
%[text] ### Important - further choose either OPTION 1 or OPTION 2, do not execute both
%[text] 
%[text] **OPTION 1** Padding - uniform length - repeat or truncate signal
%[text] For each data sample: if length \> window, then truncate; if length \< window, then repeat
%[text] Simple, straight-forward approach, long sampels are simply truncated, each data sample results in 1 sample in the training set.
%[text] Note: in our case the length of samples is set to 250 (2.5 s at 100 Hz)
%sigpad = padsequences(sigdata,1,Length=400);

%% Repeat/Truncate each signal to padLength (no zero padding)
% Assumptions:
% - padLength is a positive integer
% - sigdata is a cell array; each cell is an N×11 numeric array (time in rows)

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
    sigdatapad{k} = y;                         % N=padLength, 11 columns
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
%[output:039d6669]
%   data: {"dataType":"matrix","outputData":{"columns":1,"header":"1301×1 cell array","name":"fname","rows":1301,"type":"cell","value":[["'C:\\Users\\valis\\OneDrive\\Documents\\MATLAB\\Plush\\Training Data - labelled\\At rest\\At rest_1755548315423_666.csv'"],["'C:\\Users\\valis\\OneDrive\\Documents\\MATLAB\\Plush\\Training Data - labelled\\At rest\\At rest_1755548326428_543.csv'"],["'C:\\Users\\valis\\OneDrive\\Documents\\MATLAB\\Plush\\Training Data - labelled\\At rest\\At rest_1755548333234_328.csv'"],["'C:\\Users\\valis\\OneDrive\\Documents\\MATLAB\\Plush\\Training Data - labelled\\At rest\\At rest_1755548345438_572.csv'"],["'C:\\Users\\valis\\OneDrive\\Documents\\MATLAB\\Plush\\Training Data - labelled\\At rest\\At rest_1755548353394_503.csv'"],["'C:\\Users\\valis\\OneDrive\\Documents\\MATLAB\\Plush\\Training Data - labelled\\At rest\\At rest_1755548362012_590.csv'"],["'C:\\Users\\valis\\OneDrive\\Documents\\MATLAB\\Plush\\Training Data - labelled\\At rest\\At rest_1755548377845_568.csv'"],["'C:\\Users\\valis\\OneDrive\\Documents\\MATLAB\\Plush\\Training Data - labelled\\At rest\\At rest_1755548386440_530.csv'"],["'C:\\Users\\valis\\OneDrive\\Documents\\MATLAB\\Plush\\Training Data - labelled\\At rest\\At rest_1755548395966_551.csv'"],["'C:\\Users\\valis\\OneDrive\\Documents\\MATLAB\\Plush\\Training Data - labelled\\At rest\\At rest_1755548410407_492.csv'"]]}}
%---
%[output:68846a2b]
%   data: {"dataType":"text","outputData":{"text":"\n<strong>labels<\/strong>: 1301×1 categorical\n\n     <strong>At rest<\/strong>             51 \n     <strong>Back stroke 1<\/strong>       72 \n     <strong>Back stroke 2<\/strong>       73 \n     <strong>Ear scratch 1<\/strong>       73 \n     <strong>Ear scratch 2<\/strong>       75 \n     <strong>Head stroke 1<\/strong>       74 \n     <strong>Head stroke 2<\/strong>       74 \n     <strong>Hold in hands<\/strong>       73 \n     <strong>Hold strong<\/strong>         74 \n     <strong>Neck scratch 1<\/strong>      73 \n     <strong>Neck scratch 2<\/strong>      75 \n     <strong>Tail hit 1<\/strong>          72 \n     <strong>Tail hit 2<\/strong>          72 \n     <strong>Tail pull<\/strong>           73 \n     <strong>Tail scratch 1<\/strong>      74 \n     <strong>Tail scratch 2<\/strong>      74 \n     <strong>Tail tap 1<\/strong>          74 \n     <strong>Tail tap 2<\/strong>          75 \n     <strong><undefined><\/strong>          0 \n","truncated":false}}
%---
%[output:5fc675f4]
%   data: {"dataType":"tabular","outputData":{"columns":1,"header":"3×1 cell array","name":"idx","rows":3,"type":"cell","value":[["912×1 double"],["194×1 double"],["195×1 double"]]}}
%---
%[output:90cd383f]
%   data: {"dataType":"image","outputData":{"dataUri":"data:image\/png;base64,iVBORw0KGgoAAAANSUhEUgAAAZIAAADzCAYAAACsXZCxAAAAAXNSR0IArs4c6QAAIABJREFUeF7tnc3LHMt1xvvuot21hDbaWP4QZBGQk2yEbC8cYrIIujHZ6ANCEEJckoAEtpBk\/wHWB3JAmMRoIbQI6GMTgrUyAgdyE6GVY0EWASWyVtoIy8IbeRFQODP3mTlz3uququ7qMzXdzwvmWjPdXd1PnTq\/OudU9Xz04cOHDw3\/qAAVoAJUgAr0VOAjgqSncjyNClABKkAFFgoQJDQEKkAFqAAVGKQAQTJIPp5MBagAFaACBAltgApQASpABQYpQJAMko8nUwEqQAWoAEFCG6ACVIAKUIFBChAkg+TjyVSAClABKkCQ0AaoABWgAlRgkAIEySD5eDIVoAJUgAoQJLQBKkAFqAAVGKRAMZDcuHGjOXz4cHPy5MnVDb148aI5e\/Zs8\/r169Vnhw4dau7du9ccOXJk8dmzZ8+aM2fOrL6\/f\/9+c+zYsdW\/379\/31y9erV5\/Pjx4rMTJ040169fb\/bt2zfowXkyFaACVIAKlFGgCEgEInfu3GmuXbu2ARKBhHx39+7dZv\/+\/XvuGKC5devWAh5y\/KVLlzZAI+cLiAQe8idQERhduXKljAK8ChWgAlSACgxSYBBI3r5925w7d6558+ZNc\/DgwebUqVMbIHn06FHz9OnT1ghCICF\/Ggr6MwHN5cuXm5s3b64imNBngxTgyVSAClABKjBIgUEgEVC8evWquXDhwiJSOH78+AZIQqDA3SJlZc+RqOTBgwcL+Dx\/\/nxPRIPzTp8+vZECG6QCT6YCVIAKUIHeCgwCSRcU4PDfvXvXfPbZZ4tDdX0E0YxEI7omotNhT5482RPRtAGotwI8kQpQASpABQYpMBpIAAqBBFJXugZy4MCBRVqMIBnUfzyZClABKrB1BUYDSejJEE1IZHL+\/PniIJHVXwIr\/lEBKkAFpqiATMxlZWttf64gkYdH3aStrjKkRvLlL3+5efnyZW0a834KKcD+LSRkpZdh\/8Y7plaNRgNJaHWVrW+UXrVVq8hx8+ARKQqwf1NU2t1j2L\/xvqtVo9FAotNYqJHIKq+HDx+u9pWU3kdSq8hx8+ARKQrcvn27uXjxYsqhPGYHFWD\/xjutVh83GkhEErsr\/ejRo3s2J5bc2V6ryHHz4BFUgApQgbgCtfq4IiCJP77PEbWK7PP0bIUKUIGpK1CrjyNIpm55fD4qQAUmowBB4tCVtYrs8OhsggpQgRkoUKuPY0QyA+PjI1IBKjANBQgSh36sVWSHR2cTVIAKzECBWn0cI5IZGB8fkQpQgWkoQJA49GOtIjs8OpugAlRgBgrU6uMYkczA+PiIVIAKTEMBgsShH2sV2eHR2QQVoAIzUKBWH8eIZAbGx0ekAlRgGgoQJA79WKvIDo\/OJqgAFZiBArX6OEYkMzA+PiIVoALTUIAgcejHWkV2eHQ2QQWowAwUqNXHMSKZgfHxEakAFZiGAgSJQz\/WKrLDo7MJKkAFZqBArT6OEckMjI+PSAWowDQUIEgc+rFWkR0enU1QASowAwVq9XGMSGZgfHxEKkAFpqEAQeLQj7WK7PDobIIKUIEZKFCrj2NEMgPj4yNSASowDQUIEod+rFVkh0dnE1SACsxAgVp9HCOSGRgfH5EKUIFpKECQOPRjrSI7PDqboAJUYAYK1OrjGJHMwPj4iFSACkxDAYLEoR9rFdnh0dkEFaACM1CgVh\/HiGQGxsdHpAJUYBoKECQO\/ViryA6PziaoABWYgQK1+jhGJDMwPj4iFaAC01CAIHHox1pFdnh0NkEFqMAMFKjVxzEimYHx8RGpwJwU+I\/\/fdd8\/SsfT\/KRCRKHbq1VZIdHZxNUgAp8rgBB4m8KjEj8NWeLVIAKjKgAQTKiuC2XJkj8NWeLVIAKjKgAQTKiuASJv7hskQpQAX8FCBJ\/zRmR+GvOFqkAFRhRAU+QeLYlktVaByZIRjRoXpoKUAF\/BTydu2dbBImTLdVKa6fHZzNUgAo0TePp3D3bIkiczJsgcRKazVCBihXwdO6ebREkTkZHkDgJzWaoQMUKeDp3z7YIEiejI0ichGYzVKBiBTydu2dbBImT0REkTkKzGSpQsQKezt2zLYLEyegIEieh2QwVqFgBT+fu2RZB4mR0BImT0GyGClSsgKdz92yLIHEyOoLESWg2QwUqVsDTuXu2RZA4GR1B4iQ0m6ECFSvg6dw92yJInIyOIHESms1QgYoV8HTunm0RJE5GR5A4Cc1mqEDFCng6d8+2CBInoyNInIRmM1SgYgU8nbtnWwSJk9ERJE5CsxkqULECns7dsy2CxMnoCBInodkMFahYAU\/n7tkWQeJkdASJk9BshgpUrICnc\/dsiyBxMjqCxEloNkMFKlbA07l7tkWQOBkdQeIkNJuhAhUr4OncPdsiSJyMjiDpJ7T3YOh3lzyLCqQp4GnPnm0RJGn9P\/gogqSfhN6Dod9d8iwqkKaApz17tjULkNy4caM5fPhwc\/LkyY3els\/v3Lmz+Ozo0aPN3bt3m\/3796+OefbsWXPmzJnVv+\/fv98cO3Zs9e\/37983V69ebR4\/frz47MSJE83169ebffv27bEqgiRtoNmjvAdDv7vkWVQgTQFPe\/Zsa\/IgASyuXbu2AZJHjx41Dx8+XMFDjnv9+vUKBC9evGjOnj3b3Lp1awEPgcqlS5eae\/fuNUeOHFlYjT5H\/i1QOXToUHPlyhWCJG1cRY\/yHgzRG+IBVGCAAp727NnWZEHy9u3b5ty5c82bN2+agwcPNqdOnVqBBN+Jw0eEYT8TSMifhoL+TEBz+fLl5ubNmyuwhD6DzTEi6Tf6vAdDv7vkWVQgTQFPe\/Zsa7IgkYjj1atXzYULFxaRwvHjx1cgaXP4SIF98skne84RoSQqefDgwSJqef78+SIi0ekwpLpOnz69kQKrWeQ089\/eUd6DYXtPypbnoICnPXu2VbOP++jDhw8fhhoXnLsGiQDBQgCpKvnv+fPnF9GMjlgAEpz35MmT5unTpxs1kVBbjEiG9aD3YBh2tzybCnQr4GnPnm0RJKq4jtQVQVKPO\/AeDPU8Oe9kigp42rNnWwSJI0j0wLh48WIj\/+Of\/wzOe4Cxj6kAFPC0vbHbun37diP\/038vX76srrNHS22xRlJdX7fe0BiDYYxr7o6ivNNtKuBpe55tzTIi4aqtbQ6lvLbHGAxjXDPvqXj0XBXwtD3PtmYJEnlo7iPZjaE8xmAY45q7oSbvctsKeNqeZ1uzBYk8OHe2b3tYxdsfYzCMcc34k\/AIKtA0nrbn2dbkQVKL8XJDYr+eGGMwjHHNfk\/Hs+amgKftebZFkDhZMkHST+gxBsMY1+z3dDxrbgp42p5nWwSJkyUTJP2EHmMwjHHNfk\/Hs+amgKftebZFkDhZMkHST+gxBsMY1+z3dDxrbgp42p5nWwSJkyUTJP2EDg2GoQNk6Pn9noRnUQEW27dhA0U2JG7jxkNtEiT9eoIg6acbz6pTAc9JjGdbjEic7I0g6Sc0QdJPN55VpwKezt2zLYLEyd4Ikn5Ct4FErvb1r3zc66LeA6zXTfKkSSrgaXuebREkTuZKkPQTmiDppxvPqlMBT+fu2RZB4mRvBEk\/oQmSfrrxrDoV8HTunm0RJE72RpD0E5og6acbz6pTAU\/n7tkWQeJkbwRJP6EJkn66jX2Wt5Ma+3m8ru+pm2dbBImTBREk\/YQmSPrpNvZZ3k5q7Ofxur6nbp5tESROFkSQ9BOaIOmn29hneTupsZ\/H6\/qeunm2RZA4WdCUQFLSQGPXIkicDDSzmVi\/ZV5uNod76ubZFkHiZMIESVjomLETJOMaaEz\/ttb7njfu09R\/dU\/dPNsiSJxsjyAhSJxMLauZvs6m73lZNzfBgz1182yLIHEyVoIkDpK26MPuYJfj5I8724cbb19n0\/e84Xe821fw1M2zLYLEyS4JEoLEydSymunrbPqel3VzEzzYUzfPtggSJ2MlSHYbJN6D0skse\/+G+FT1GFt3T9082yJIxracz69PkBAkTqaW1UxfZ9P3vKybm+DBnrp5tkWQOBkrQUKQOJlaVjN9nU3f87JuboIHe+rm2RZB4mSsBMl2QJJawI+ZgfegjN1Pqe\/7Plff80rd965ex1M3z7YIEieLJEgIEidTy2qmr7Ppe17WzU3wYE\/dPNsiSJyMlSAhSJxMLauZvs6m73lZNzfBgz1182yLIHEy1jmBJMeA9bGpaSg5Tv5S9pGkXrPLDGLt5Tyvk7klN9P33vuel3xjEz3QUzfPtggSJ4MlSHY3IiFI9vadt5NyGqZFmunSxlM3z7YIkiKmE78IQVIfSFIHGkGy7jtolqpdfGRM7wiCpK4+\/ejDhw8f6rql\/ndDkAwHiXZiJVJbqc6QICFIckY+QZKj1vjHEiQFNU51milNxq4V+163kVMjIUhSeifvmJy+kiszIonrS5DENfI8giApqHauw+hqOnat2PcEScGOHXipnL4iSNLEHgskffsq7a6HH1Vr1oUgGd63qyvkGiFBspnO6UqlldS2YJcnXSr33hmRxGUlSOIaeR5BkBRUO9dh5IDEXjunrW2ktm787FVz5c8OJ7+wMFQjid13wa4b9VI5fcWIJK0rCJI0nbyOIkgKKl3S8cXAkeOcYvcV+j7k2NukCt0LQbIZbaXsx8EZjEiWSvSFRc7YsDade27u8UPdDVNbQxVMOH\/bIsccdsIjtKbJYmBJjW5Chp8CktigtmkpgoQgybH30LExm2uD8xDnnntu7vFDNdm2j2u7f0YkQ3tWnb\/rINFRiI1IYoOaIGk3pFxnw4gkLyIZMsliRFLGARIkLTqmpHa6DDjmPHK\/D7VlnXdK6qkrIiFIygyqUs6pzUZitjPOU\/hfNTZ5QUSyCyAp1WeMSBzssKTIBMnmu7Zig9pC7ZN\/\/GXz07\/9GovtkVx\/V0qHIHnX+q63ruh\/iNPOPTf1+NTjYm6ypI+LtZXzPSOSiiKSnMGRAjo8WizlplMpAMLQ1BZBMl6NpJRTynEU2zg2NnmZYkQS61uCxMESS4rc5ajbctgxh60l6EoxyXGxcJ0gcTCozCaGRBAh2xlyvcxbr\/JwgmRvt5T0cSU7fXIRyT89+UXSq89jIpYESQwaAIdOD7WBJBQ9pDxL2+xNQ0s\/s31+rMJqS8X8+\/+8W+wbwZ+OSGzaq+0a9rgcMMc08Ph+iOPPBUmKph7PPEYbbRO1tolYbNKVc4+xiMBeK\/X4UscRJDm92fNYEXkoSFIcdW5Esqsg0XAgSNZG2eYU2jRKcSIhkHRdjyBZ10+2CRLdR6kRFCwpxS9YV0iQ9IRDzmlTBondl+GR2iJIwtbX5uBjIElxNLpfCZLyxfYY1GPfW4sgSJaKMLVl6hG5EUlbOiqWlgnNorpSW2OBpC2dJZ+ngATn90ltWY2GprZCjjfXMaRMXAiSFJX6HzNmaitmD7HvCZJwvxIkLSARZ\/GNr368qre05bDHAol1VnMFCfRNGeBTBkkMuv3ddn1nDgVJLA1rd8THJn1dCjEiYUSyso+2\/LQGiXbk2rmNCZK29mUglEptjRGRyDVFL+wjacvnt7Utx+O7uYFET2DanFRO39eHifgdpYBE62QnGV3OPVaXSJmw6CcgSAiSXiCxBlwSJDaV1AckoUHYNeMKOXMI0ze1pUFiIzs9CHcZJLpv8EwlaiQEyd4f9go59zmCRHT4q2\/\/UfPy5cs4jZ2PYGorMbUFJ0GQbFpoqEYyJ5BoJ1caJFhCjQhNL+G2NSlnvzFqc3YyRJAs5SZIRjW79cX7rtoKhad2Jo1jZHDLfolQeqlt5h8Lp2EktUUkSE\/J\/cVWEOl7J0j2whb2EntjrbY7DRJrowTJuoZZa2orVntJ8Qs2gmdE4gATgKQtJ992C54gaUs9yb1NBSSipzyn1Ei6UhBTSG21RSQhJ9KVf9eRHdJmXSDRkHcYWq5NhBaWaACLVjJWoFMqSLS9WQfdtWG36+FT6zGp0IjZCEHiYIpTAYkeKIBi6qqtvjWSUPec+If\/bN7+\/bcWX+VEJKVBgnuL\/ThU6B5Degw1xTY4dhXHYxEJrqkdZApI2vpl6DNu83ydRg79yibGh9yjwMROHEP9AIiEJpmxyIEgiVvDJGskux6ReIFEDy5ERBiY8u9tgAR9Z+8tpU+3CRL0GV4TE4tIQrNo0T4GEuiCiGSOIJFnxh9BEnfyHkdMDiSf\/uTnG\/s\/UkTEbFDPfMeokdhUjg6nrSOHQ8mJSKzzDYXr1sFpfXSbY4PEzhC7\/l1rRGJtC6nJNqi0pcEQ8REky4IydLRROPQGSORYgFtHq\/q80MTEjjs9SelKLYV8CVNbS1UmCRJ5MP0CwRhMQmmFPiBJSW1owx4CEnt\/2hlZx9sWumPhAACyCyDpGujeEUkqSEIO0d5rSmpLR4tTjUhSQCJ2i1ShBomtM+lxEEpttX2WUy\/R\/Wj7NJYys7asJ5ohnyXfs0YS8+YFvpcaiUQkJUCiV2fBSYvR7v\/uvzaP\/+4Pg6u2vEFinxMzNRvut6VZ5HgMGj0bRleMldoKRU6pEUkuSNpmtUPMzUawuJaeSctnKAbbnL+2Jz3LjkUkFiRij1NbvZUKEuimQWJhPBQkXdGG7jfcg\/gG1BTl+1D0bxcN2H\/r6MjaKEEyZNRmnKtBEtow1napttkglvnWDBK7aREOTEc+2jhh3PLfbYBE7sVGPnoptb1v3WcpBWsbidYKktBEhSDZm9oK1UBEu6Eg6UoDwwYtSEJOXh8zFCShLIO2\/9mC5MWLF83Zs2eb169fr\/Q4dOhQc+\/evebIkSOLz549e9acOXNm9f39+\/ebY8eOrf79\/v375urVq83jx48Xn504caK5fv16s2\/fvj1cEJD8waV\/Xsyy296TFYLJHEEiz6xnTF4RCUFyeKE7HBBmpDbNKJ\/bVVtzjEhSQaLtyqaYYhMVuwcKwLAbQscCCcYhQdIy1RdI3Lhxo7l7926zf\/\/+PUcBNLdu3VrAQ46\/dOnSBmjkfAGRwEP+BCoCoytXriSBJJZ3RLQRmg22RSRX\/uxLixqMvfYYqa22wqNOY8EZhVJb9h610W4LJKGBrmGuta09IpH+QRSYmtrS0SAiqFSQIJqDhlNPbdnIQ\/4dikhSQaIXROAc+a9ebYc2UYeSf+uUm7bJEhGJre10bVqdZY3k0aNHzdOnT1sjCIHEohMVFPRnAprLly83N2\/eXEUwoc\/QsYhIYBipLzfMjUgAEjuDaAOJTq9gVhPKn+qUjy6Ai5HbFE0IJChC6llcDCS4HxuRSPs3fvarjX0kesBhcIUGI6Id5PD1xjGAW6egpJ+mChLoY\/tQ6k+wIzhHqxfgojUjSMqARNubBQkmBzJ2ZFPtWCDRY1pPSAgSEyOEQIFDkLI6fvx4c\/LkydWZEpU8ePBgAZ\/nz5\/viWhw3unTpzdSYHKBsUEiDlE7gFyQ6ONjILEzzrmCxC6D7iq261SEnlWGNrUZU836J\/pRQxQgBjDkPgHLUFRJkIQlt8V2HKUnHlKL+MZXv7CchKqfd7aQxbl6Qqn7An1GkGSZf\/Dg0Zb\/wuG\/e\/eu+eyzzxaN6\/rI27dvm3Pnzi2iEV0T0emwJ0+e7Ilo2gA0BZCIg0J9R4NEOymdmpJn1rN9RCl2SaTNASMCWA7a3yxmxjkRiS7YjxGR6FrALoAE0NBOygskOv0y3B1s\/wohkEBLzNQ1SDQEckCiU4n6+mhfonGBVWpEIudhlaMu5Nsl\/m37XRiRtNgeQCGQQOpK10AOHDgwCkj07Vy8eLH53e\/\/xZ5lmPoYGI52pNKpcJZwypiFxiKS0Pt\/AIVQRKKNrgskcBg4Xh+LGVdJkMi1BDL6FSkYtDZVh0GgtdLONTe1lQMSHaF4RSQ65QRb0UtxbWpP64ZnKxWRECTrqKQNJDZlLH2FPtRjBuMIad02kGib09HpGCC5fft2I\/\/Tf7N\/jTyiCYlMzp8\/PwpIZNUWZtlIM9j1\/BYkodl\/Lkg0kPQyVTvDsqBJBYk2Uhg6IgkLEr1qzcJFRzRdEclUQAK4dKXEcufhOSAB8GGLiCK7QALwdNVI5HzZz0SQrEGiJ2yhlJfoCr1yQaLHkSdIcM+YgMyy2B4aoKibXLhwYbECa4waiRdI9F4AGFoIFDYva0EDndoikqUxLQvfNvTGyjL5HPcAkITSTmOBRK6LNICdkW8zItHOuK2IORZIdI5egwT9qRdt6AiuDSR68+jcQCLPrjf+oUaiU1saJKExIHZYAiRoB05+PX7XY1T3vfx\/PaHU493aRWj\/m456ZgeS0OoqW98Ya9WWzvvbGaFe5qdrBXrFjE1t6WhDzyT7gkQbmTYSC5LNfOrSSAEL5HDHAAmePye1pUGC5Zl9V23Z1BZm8aENiXpQh1JbeBOBXGMMkOiZqq2RlAKJtkcd0cwpIkkFiS6s68UO0M0bJLA7giR3uvb58TqNhRqJLAd++PDhal\/JWPtINEis49fFsxIggYHHIpJQMVun3vTsBp+PCRI4+1CxvQsktvCv7xvapoBEzxjl\/9t9FDA7W4S0KaopgUT3t9gAIGhBoiMaDfCeQ7Wq09qK7WOCRMaAfu1RqEaiJwyY\/MF\/rCOOZU1Rp6tTQQK7ZkQSMEe7K\/3o0aN7NieOsbM9BBId1qLjQyDBY8AYxDl2RSQwcL2HI1Qj0UsN7VJRXYCVkF2Hu3AuSG0hbSTPiGNtjUSH+3BCFqAeINGpGD1T1JEXBlppkGBgames8+ZDvKetXUDvWEQCG2lLbREke1+RoscqnGzbqi3YFcZuLCJZF9fXILERv03X2h9rw\/GoNxIkQ0ZWJediH8lYILEOAI4Bq0D0LNvORLpAotMVpUCCe9Eg0TMlidSkLRuRaJCGUluhiATXshEJNAjVSDTsuiISDZ2U1JZO9cnx2wQJYG8nG3qRg953op+PEclyE+5QkCCiDaW2vECio2s9mdE1HR2RYNzgvFnXSLbBljaQiMOE0cRSW10RSRtIEO1sEySIVHD\/OSABvPSgkwhI\/rD8V0dcmHkDTKkgwYDQaYJYRNIHJFqDLpD0XckVikgQTWgwxECC5aXyjKK3FN\/1asG21JYG0a6mttq070pt9YlIYiCBneuFDxjnOuJHH0kKDMuJU1NbBMk2aDCgzRSQ2PSG3QewXtW0dqQw7hBIdOSRCxIbpcAh90lttYEEThspOsirI5IukCB3rNMyiCZSQYLj9ZJL5KXnABJ5RtFfwABYADJIlRAkS8tMBckywljubgfEMXZ1akvXl+yqrSUIluN8GyDB4hCMXZsKZkQyAAZDTu0CCa6r86Zw5HrpZW0gwUBpq5EgbZIKEsymhoAEs2ALEtwLcthImxEk6SABUGxqC5+vne3yjQRTjEiwsqortRUCiYYxJip2oQL0wsRoTJDohSJyv4jk4YsIkiHefsRzu0Cii9Po1LFAokNfXajX0Q\/yo9rYQhEJnPEYIEFX5EYkbSDRM29dfykBEh0F6aK0XbWlI85YjcQrtWV1WdrfcrGEjUgsSAQW8jcVkOioQfcjbFG+tyDBs+vUVg5IrHaiuV5sUjoiQQq3CyRIlWFJv7ZFu3KLNZIRoRG6dC5ItPFiz0OJiES\/kiEELdRr0OZyUCxfCGmL7XA4KHzDADGQhkQkQ0ECA8d9h0CC+pR9LUVuaqsESOwA3QWQIH2jozvUSPAd6n7Ow61Xc7WCREMd4Na221UjQX\/IeQCDTVvDD8COc0Cia3Kz25DYy8oGntQGEusw0alIzaBzdRisi825NRKAxP4XofoQkIhRiVPRMzLMarVBWyCFoBUCiQWprZFAM70MMhckcrz8ybXlL2XV1i6BRM92tbPXy1ZTI5I5g0SnnxCZIeXXNyJBZG8jEoBEt0mQpDvk0d7+m34L5Y4cCyQAAGYqKMzBGPEEeqUUCnt6d\/c2QaLTCIh+UkEix2FjZy5IlgBY\/hAYZla5IAFs5Bpof710c\/26dj1zwzl65dOYEYm+x1yQYHIgOuk0jga0jkgwyZl6RJICEquLXsyAFVZ4C8RyArncNJgCEm27uRGJ9mrYJ4ZUFxbT2HS13UeEa+i3MzAiKceL1ivlgkQbWFdqqwskmFnCkegd7Fi6upw9Ld\/zg+PaCvyx1FbfiKRmkCwB\/WoBG5sW1Jr1BQlms2OCRL\/frTRIEIHqeomNSB2GV5EmclJbVke9mAPPnwMS6EiQFOnKjYvMKiKRJ0c6RRyUrj\/o2QtmhnK8fr9VKCKxaSbMNuxrTjRI0AMheIVAguMxAEKprdAMyy4jxiAuFZHoCCOUwtEOEBEJZoW6LzRIdPEcqcZQRAJN9EqYtohkGyAJ9VFbaisWkRAky1QuIlvRUf\/ZBQlI8dqIpARIrC1htaS+tp54YhK5tPvNn2WORSQYr4xIyoOv84qxiMQTJHYXNkGyTG1NGSQ6z65BgnRKDCT2TQN6iTYcKSYzc4lIoGMqSHRNaVdBoldF6pWHogFTWw5QqQkkeFy9ycjOVIZGJHAmoeWMOrIJpd9Qp8A12lJ7iOBsvh7RkZzXVWzXDqANJKFl0ZjB6ZqITm3VGJEMBck6Ul3+YiVBsl5YUiNItI3mRiTIiISW\/xIkDrDoaqIPSHC9WGpLr2bSxeO21Bacdy5IEDUBOnpGZlNbUwFJqHa0nHkva0ryh\/y4Xmmni5dyzLZTWwRJ3AHk1kjGjEj0tUOrtjS8UGzXqS2CZN3fs6+RpIBEr86yRVVrjKhLzAUkeM0E0jICO72z3c4kcRyAaYGxPP7wolt0pDMEJFjGOUax3UYNoTpWamqLEcnyFSnYkKgXlowRkQwByXIS9\/Gi7hHaR6KRqldtYZFNKCLBhCkUkQBgd\/7mT5rZ\/9RufL4y7IhSEYl2dnqFx3IGsnw1BYrHMZDAOFJrJLsWkdQKEhThATUszdar17o2JHZ9pyOfGEhg0RqwsKNQBAOHydTWcve5V0QiumOst72DS++QLw0SrCBFbRXvxkONhCAZxoass8dfYWYwAAAQGklEQVQGSWgVUk0gybmXUjWSMUACpz8kIiFIsoaO28Gh1Jb+DBEq6n61gQTRLUBiU9B2lSeEta9jshFJaK+LXrVFkLiZaNPMGSR2SaRXsV3P3jCrG5raQtSgB1dKaksvk0R6JCci0VFIW0SC1AsWJ2BzoI487Eza6jK3iCSkq17mvcsgCUWkdlGNjV50+hvgaQMJUrwECUGyKBTr1BZmWSjwh15LkltsLwUSGLS+R1uvQM7fvkGVIFmnR\/RMeq4g6Yo++oLE7oFCKnAZyfxq9Wp50T+2\/DfUR11vBbYTm1D6sWREAh+g016skTgARSKSd9+5u2hJCB4yOrshEbcVMzrMInHtITUSgmS9wSxUO+oTkYjTktmhhjN+AyS12L5rEclyprt8g3CNf+sJ0vKtBfLX9RlW5HUV21NAYseXnRyFVj9q2OsNyfYNzdinYtOvXct\/+0YkBMmWrDoHJLpGILc7BkggA6Bmo4xYRNI2YwqtNvGKSPSASY1IMBhDKS8LEixmsDPMWGqrFEj0rz\/qwjz6sqbUVu0gQdSho4+uz6YEEjspsfu0sNKrK7VFkFQOEnEWNrcdA0mo6Gdz4bYukQMSHfHIvaAYp6Xs2kdCkKwjEgxARCRwuDKYu1ZtLQf38iWQY9ZI9N6jtqJybOYNu6g5IiFIPl5FYQTJlqDQp9nUiGROIEHkFQrJtTPSxWO71wN6hZZFpxTbdUSi+9VGaoCpfneSvm+7sx1r+LFUMvSGYgAWkCdI+oysfucQJHkgsZNGG5HI9\/916y+5j6SfOaaftcsg0RFPyYhk6iDBi++Q2sJrxMVqUiMSXRTuikh06qXvqi1tzSn7SCx4kfJjRLJWxr4VGd\/kZBnkHOz1KVEj6Upt4f6waAW75kMg0ZsdxV4IknQe9D6SIFlLhxl4H5DoDogNRp2S04NRRy8lIhKAAc582e4yRSkOgCDpPWw2TuzaiJnbgo1IdOpQ0sJYJCDHYff3EpSvFt+FNiSmpvzkOjHbbVtZFwIJ7iVWbNd1T7kHbDDUqS2CJNeSnI8fChLUJUIpnJSNUUNqJGNHJNIVevVS24bEIbWjISCx+mLmjYFrQYL3bOlXcWPnsS5kpqS2uiISO7OEoysZkYRei57qMNtqJH2BkHpeynGABYCSC5IhugwBSWgSRJB0O\/PJvWsrZflvW42EIBm2CIEgie8j0cMxFqnZlYWhFM4uggQr9WIRybZAEuoj+4Nybct\/UyISvIePqS3nSCO1OUYk7amtrojEzt5yUltt6YFSDjMUkSAaQdSil1Dqn1pFzhvP3rZqayoRid0hjoUFseghZf+MHYOxa8rxbRHJNkCi7z+2j4QgSfW46+MYkXyuhS5wj5Xasq9T0LOXbaa2UkEydDD2mWFqkGBTm\/7lRL3LHntQkJ6LgaQrh6+dsnbINae2Qikkud+Y0w\/tMkdfdy2DDu2z0TaCl1vqBQz4KWXUQaCn\/YlqvVAC12zbZJxSbB9qu30jkvW9L39GGmlX+JguqIlGLLbnQ23wGbVHJATJ3p9IjdUC2kCidyvrwWh\/JrmrRqLfiYSlxXbVlp2t6x\/hCqVI+8x2+wAWg0WntkqBRK4di2ZicEJEglqS1hcRCTaZotiOSFOuXRtIuvootFwei0KQxrI1tVSQ6JolV20NRkTaBQiScVNbQ2d1fRwmBhwWMsg92H0tKc7b7iNB4VfPGOUzvWIIEZB2qrWBRO5fnl9HSrqo3SciGQoSQCYWkQAkOE60169R72Mv2kb1qq0xbTcGklDET5Ck+fStHEWQTB8kerVWymBsi0gIkrWt2NRWCZDYFCAiErwPTb99YKogselqHcGm2K7dX8KIxAkrBMm8QIKnTYlI5Fj8IBlm6bpoD0c3xYhEg0IPxXX6ZflaGOhCkGyOo1iarS0iIUicHH\/pZnYVJKG9G6XetWX3i+hliylpAK\/0gL4X+4oUm2deznZ\/tTqlD0h0fUQuhA1xuwoSnc7CswEOfUEC0LS9vLKt2I7z0K6uO00xIgmNI10kZ2qrtKcf+Xq7AJJU502QrH9fwr4WvCRIkM+PgUSnwuT\/17IhUbTAzn447DaQhDZXyvk4T0ck2k4Jkq81qfWarpWYJRZn8BUpI0NELu8FkraNZLGd7V2vrreAIUjGAYkAY\/miyVerVA6iNJ3X17N4FID1qqIaQYJoKgQSgNIuZU4FiV2l1bVqixHJepkvxnWpyJ4gIUgWP7zEiGStQMq+gFD+eUhEIufip04xA4+BBHdMkCzfZqt1Cw1rgET\/0Fis2A5Y7+KqrdQxzYjEAQIlmmBEslYR0dFUaiTyZJjVDQEJ9ihg1o73c+nUFkCjAYOZO9ouNcOMFXL1uGjbeKdfWqk10jWSvhGJrm\/IvcT2lwAycJq4n1SQoB5WQpea+qjUvTAiKUGKyDUIknmAxBYyc4rtGiS4DiIjm9pC9ILd2HrpcSnHUMJhhkCC50RtBzvH7eosADL0uf7OFt5DqS18hjYF0qKTTGbsPdrlv+v04W9WPx+cWpfw2Nley70QJAQJU1vf7bezPfQb93ppZSpIsEtenLd+\/YQGCd4+AHNFLQSOGPCpFSRwqhYk+nkQNeiUXQpIbFFep6N0ukuDBG8aSAEJVuKVeJmljmDtqshUe5Fr1HYvBAlBQpDsMEhgvrsCEv07LZj9Yyd5CCSIvhB5WMi07Za3K8H0wgSkDe1vxuC9dtIGIrLQb4DUEgWUiBpLTTwIEoKEIOkBEm02bUsrU2eYQyKSzbTNbzZ+NEnfY+q9lJzt2h\/2Qh0Hn5cCiX7RotbD1k10RKJ3cGOndggkod8AIUg2I3jubHeAiDRRS40k9DsSoZcTts1SxvipXTiuXduQ6AES\/R4v\/WJNfI73a+k8fqkZZonZrgYJYIm3IotT7wMSXReyu\/51vcT+fxuRECRLCy5lL4xIHGBCkEyv2D4WSOT9Tsti8rK4ixpB2y9H7kpEMiZIUDTH23p132Bvjn7Lr337ACIS\/GaMTm3pa9VWlygBe4LEAQClmiBICBKbZtLpEZ3amjJIsBBBp9nk2W2NxK5G0xDQu\/exHFr\/rDGWE+MaSyi\/23jVjF0QAZCgHwiS\/F\/UZERSihYd1yFICJIckMDRdUUk4Z3gddVIZLYrf\/b3bmIg0e\/CkvPtRk39WyKAAH6N0u67sa\/Xjy3R1ikvWwthRLIETEgXgmSHQBIrnsZekcIayd4iYcn0QMxJof\/sYLQRyRxBglfEoHYCWKSCBCmaEEiwb0T\/EqDuC\/srg6jjECS\/XKRYuyZB8h2L7Q4QkSZKRSQEyVqBUrnd2kBiU15tixBCbyIWdWrTJTUiiYEEKT+5HqIx7KHRoNAgwbH6+ynt3Shpu0N1YUTiABOChKmt1NRWn6Wlofd+DXUMpZxUCkgQdWBlFX6REJpJdEKQjBtND7UXgoQg4T6SivaR9AGJNuFaI5K2aBppPYEFVljZ15xYkCDS0BEJIjQbkcixkp6pTZcSL0osBfsS90KQECQECUGyMQo8i8ohkMjsGKu5EK3oiETAgNfH4GcNNEhQC8FDESTLX+GUv9CP1REkDhAo0QRTW92praWz+NJiiWbOK+1LDICSs7oSxfapRSSxPrIgQeEbn2NpsOii96KEQKKXButXqhAkBEkJP771axAkBMmYNZKaU1upIJFnEKgTJOkrpUpOglgj2Tom4jdAkBAkBMlaAbuPBN8QJN9agbTLXuQ7z\/Rjyr2wRhLnwOAjCJJ5gGTorK6mGabXvXTtJte73lNSW3a\/CGsk63HHGslgN779CxAkBEnKrM7Ledd0L0NBgpoaXveuV2fpmgpXbbHYvn0SBO7g\/fv3zdWrV5vHjx8vvj1x4kRz\/fr1Zt++fXuOJkgIkpqcd233onf2hzZkyvLftoika3EGQcKI5KMPHz58qJIgn9\/UjRs3mtevXy\/gIX8ClUOHDjVXrlwhSDpyuHh\/VInXtcQKubU5TNxP2\/uK5hiRxGpHdtUWXtmRusoPmzUZkTAiqY4nL168aC5fvtzcvHmzOXLkyOL+Qp\/hxhmRTDsi+dP\/+7fmj\/\/8r\/es0d\/Gj0lNzWEKMHREkgsSWN4QXX7vv3\/a\/O73P6muwF3TxIPF9h6YevbsWSMRyd27d5v9+\/cvroBU1+nTp5tjx45tXJUgmTZIPv6Xc82nP\/k5QfKzX23Y\/RDnjQtpkOiLp0YkJUAi\/fvuO3cJkoZv\/+2Bi\/ZTHj161Dx9+nSjJgKQHD9+vDl58iRB8tUvrH73WosxxdQWQTJe2iQXJGOsTiJI1iOYr5EviJJckJw5c6aRKIZ\/VIAKUIEpKiBZmPv371f3aFUX23NBUp26vCEqQAWowAwUqBokuTWSGfQXH5EKUAEqUJ0CVYMkd9VWderyhqgAFaACM1CgapCI\/jn7SGbQX3xEKkAFqEB1ClQPkpyd7dWpyxuiAlSACsxAgepBMoM+4CNSASpABXZaAYJkp7uPN08FqAAV2L4CBMn2+4B3QAWoABXYaQUmARLWUXbaBhcLKu7cubPxEJ9++unqxZwp\/StLxWVDKv5k05Z9hc5uq7R7dy+rLn\/4wx82P\/rRj1avOJKnSOlPbRNHjx7deE2SXIP9XZc9TAIkXNlVl1Hl3E3XK29wnVj\/isM6e\/Zsc+vWrQU8xMlcunSpuXfv3uplnzn3xGOHK4A+OXjw4B4IxPpTNiI\/fPhwdZ4+Xn4+gv09vH9KX2HnQcK9JqVNwvd6b9++bc6dO7eIPkIRREr\/iqORP\/3TAqHPfJ9svq0JCL7\/\/e833\/zmN5vf\/va3GyCJ9eeBAwf22IO1EfZ3fba18yDh7vf6jCrnjrp+FgApjK43QEvaQ36jxr7EU+ziwYMHrT+ClnOPPDZdAXH63\/ve95of\/OAHza9\/\/es9b++OjVcBif3pCGldbODw4cPNJ598wv5O7w63I3ceJHwfl5utjNIQHMubN28WP2Amf7o+Euvfb3\/728GIJuSwRnkAXrRVgVAfxPrzi1\/84h74ACTy3\/Pnz7O\/K7Q5gqTCTpnTLYlj+fGPf7yqZ6Bmgl\/BjDkegqReayFI6u2b0ndGkJRWlNcbrIAulv\/iF7\/o\/E0agmSw3KNdgCAZTdrqLrzzIInlXLkEtDqbi96QrpuE8uz6VzJZI4nKubUDQmMzNl5ZI9ladw1qeOdBElsFgt96H6QSTx5NgdAKHO1sBCS2+Gr7nKt4RuueQRcOQSM2Xrlqa5DkWzt550GCQpwUaq9fv74QUlbxIMe+NWXZcJICds8HlnqeOnVq9VPKsX0H3FeQJLX7QW0LHmL9yX0k7l01uMFJgCRlp+xgpXiB0RSwu5SvXbu2gog0mtK\/3Ok8Wvf0vnAbSFL6kzvbe8u+lRMnAZKtKMdGqQAVoAJUYKEAQUJDoAJUgApQgUEKECSD5OPJVIAKUAEqQJDQBqgAFaACVGCQAgTJIPl4MhWgAlSAChAktAEqQAWoABUYpABBMkg+nkwFqAAVoAIECW2AClABKkAFBinw\/10RrHzD1P10AAAAAElFTkSuQmCC","height":0,"width":0}}
%---
