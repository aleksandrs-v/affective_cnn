%[text] # Real-time CNN classifier v2
%[text] Loads a pre-trained network and classifies the incoming signals from serial port COM15.
%% Real-time CNN classifier from COM15 (lag-optimized; safe shutdown)
% - Uses a dlnetwork saved as convoPlushNet in convoPlushNet.mat (with classNames)
% - Reads CR/LF-terminated lines of 14 integers from COM15 (ignores the 14th)
% - Replaces accel X/Y/Z (cols 11–13) with magnitude in col 11 → 11 features total
% - Sliding window of 250 samples; inference every HOP samples
% - Plot updates are throttled to avoid UI lag; serial is closed cleanly on exit

function runRealTimeCNN
%% Load network + labels (keep this file in the working folder)
S = load('convoPlushNetMCU.mat','convoPlushNetMCU','classNames');
net = S.convoPlushNetMCU;                 % dlnetwork
classNames = string(S.classNames(:));  % 18×1 string

%% Parameters
comPort  = "COM7"; % Dog_R_02: COM5/6 Cat_R_02: 7/8
baudRate = 115200;    % set to your device
fs       = 100;       % Hz (for reference)
winLen   = 250;       % model window length (samples) def=250
hop      = 50;        % inference hop (samples)  -> ~100 ms @ 100 Hz
plotHop  = 50;        % update plots only every N lines to reduce lag
emaAlpha = 0.2;       % EMA on softmax (0..1)
useGPU   = true;     % set true if you have a supported GPU

%% UI (pre-create graphics; update YData only)
f = uifigure('Name','Real-time CNN','Position',[100 100 900 620]);
f.CloseRequestFcn = @appClose;  % ensure clean shutdown
gridLayout = uigridlayout(f,[3 1]); gridLayout.RowHeight = {40,'1x',180};

lbl = uilabel(gridLayout,'Text','Waiting for data...','FontSize',16,'FontWeight','bold');
lbl.Layout.Row = 1;

axSig = uiaxes(gridLayout); title(axSig,'Latest Window (feature 1)');
xlabel(axSig,'Sample'); ylabel(axSig,'Feature 1'); grid(axSig,'on'); xlim(axSig,[1 winLen]);
sigLine = plot(axSig, 1:winLen, zeros(1,winLen,'single'));  % created once
axSig.Layout.Row = 2;

axBar = uiaxes(gridLayout); grid(axBar,'on'); ylim(axBar,[0 1]);
title(axBar,'Smoothed class probabilities');
b = bar(axBar, zeros(numel(classNames),1), 0.8);  % created once
b.XData = 1:numel(classNames);   % pin bars to fixed x-positions
% Initialize
axBar.XTick = 1:numel(classNames);
axBar.XTickLabel = classNames; 
axBar.XTickLabelRotation = 45; axBar.FontSize = 12;
axBar.Layout.Row = 3;

%% Serial and shared state (in UserData for callback access)
sp = serialport(comPort, baudRate, 'Timeout', 1);
configureTerminator(sp,"CR/LF");

state.buf         = zeros(winLen, 11, 'single'); % [time×feat]
state.widx        = 0;
state.filled      = 0;
state.count       = 0;
state.hop         = hop;
state.plotHop     = plotHop;
state.net         = net;
state.class       = classNames;
state.emaProb     = [];
state.emaAlpha    = emaAlpha;
state.useGPU      = useGPU;
state.axSig       = axSig;
state.sigLine     = sigLine;
state.axBar       = axBar;
state.barHandle   = b;
state.lbl         = lbl;
state.winLen      = winLen;
state.running     = true;     % flipped false during shutdown
sp.UserData = state;

% Start streaming callback
configureCallback(sp, "terminator", @(src,evt) onLine(src));

% Ensure cleanup on errors/interrupts
cleanupObj = onCleanup(@() stopStreaming(sp)); %#ok<NASGU>

%% ---------------- nested functions ----------------
function onLine(src)
    % Guard: if shutting down, ignore callbacks
    st = src.UserData;
    if ~st.running, return; end

    % Read and parse one line quickly (14 ints; ignore 14th)
    try
        line = readline(src);           % CR/LF-terminated
        % Fast path: assume comma/semicolon/space delims
        % Replace commas/semicolons once; then sscanf ints
        %line = strrep(line, ',', ' '); % input vector is separated by spaces
        %line = strrep(line, ';', ' ');
        vals = sscanf(line, '%d');      % column vector
        if numel(vals) < 13, return; end

        % Build 11-feature row: 1..10 + magnitude(11..13)
        ax = single(vals(11)); ay = single(vals(12)); az = single(vals(13));
        mag = sqrt(ax*ax + ay*ay + az*az);
        feat = single([vals(1:10); mag]).';   % 1×11

        % Ring buffer write
        st.widx   = 1 + mod(st.widx, st.winLen);
        st.buf(st.widx, :) = feat;
        st.filled = min(st.filled + 1, st.winLen);
        st.count  = st.count + 1;

        if st.filled == st.winLen
            % Decide whether to plot and/or infer for this line
            doPlot  = (mod(st.count, st.plotHop) == 0);
            doInfer = (mod(st.count, st.hop)     == 0);

            if doPlot || doInfer
                % Extract ordered window [winLen×11] with newest at end
                idx0 = st.widx - st.winLen + 1; if idx0 <= 0, idx0 = idx0 + st.winLen; end
                if idx0 <= st.widx
                    win = st.buf(idx0:st.widx, :);
                else
                    win = [st.buf(idx0:end, :); st.buf(1:st.widx, :)];
                end
            end

            % Throttled plot update (feature 1 only)
            if doPlot && isgraphics(st.sigLine)
                st.sigLine.YData = win(:,1);
                drawnow limitrate nocallbacks
            end

            % Inference on hop
            if doInfer
                X = reshape(permute(win, [2 3 1]), [11 1 st.winLen]); % [C×B×T] = [11×1×250]
                X = dlarray(X, 'CBT');
                if st.useGPU, X = gpuArray(X); end

                Y = predict(st.net, X);
                p = gather(extractdata(Y));    % numClasses×1

                % EMA smoothing
                if isempty(st.emaProb)
                    st.emaProb = p;
                else
                    st.emaProb = (1 - st.emaAlpha) * st.emaProb + st.emaAlpha * p;
                end

                % Update UI (bar + label) without recreating graphics
                if isgraphics(st.barHandle)
                    st.barHandle.YData = st.emaProb;
                    N = numel(classNames);
                    set(axBar,'XLim',[0.5, N+0.5], 'XTick',1:N, 'XTickLabel',classNames); % resolve rescaling
                end
                [pmax, k] = max(st.emaProb);
                if isgraphics(st.lbl)
                    st.lbl.Text = sprintf('Predicted: %s  (%.2f)', st.class(k), pmax);
                end
                drawnow limitrate nocallbacks
            end
        end

        % Write back state
        src.UserData = st;

    catch ME
        % If figure/serial is closing, swallow benign errors; otherwise warn once
        if isvalid(src)
            % Optional: uncomment to diagnose
            % warning('Serial callback error: %s', ME.message);
        end
    end
end

function appClose(~,~)
    % Called when the user closes the UI window
    stopStreaming(sp);
    try, delete(f); end
end

function stopStreaming(portObj)
    % Safe shutdown: stop callback, mark not running, flush & delete serial
    if ~isempty(portObj) && isvalid(portObj)
        st = portObj.UserData;
        st.running = false;                % prevent further work in callback
        portObj.UserData = st;
        try, configureCallback(portObj, 'off'); end
        try, flush(portObj); end
        pause(0.05);                       % tiny settle time
        try, delete(portObj); end
    end
end

end  % runRealTimeCNN

%% Run it
runRealTimeCNN;


%[appendix]{"version":"1.0"}
%---
%[metadata:view]
%   data: {"layout":"onright","rightPanelPercent":27}
%---
