%[text] # Manually train and evaluate various architectures
%[text] 
%[text] Define and traing various architectures of neural networks, construct confusion matrices etc.
%[text] 
%[text] You can even try LSTM networks first:
%[text] Set training options, train the network.
opts = trainingOptions("adam", ...
    ValidationData={valdata vallabels}, ...
    Plots="training-progress", ...
    Metrics="accuracy")

layers = [
    sequenceInputLayer(11)
    lstmLayer(128,OutputMode="last")
    dropoutLayer(0.5)
    fullyConnectedLayer(18)
    softmaxLayer()];

plushnet = trainnet(traindata,trainlabels,layers,"crossentropy",opts)

%%
%[text] Check accuracy.
acc = testnet(plushnet,testdata,testlabels,"accuracy")
testscores = minibatchpredict(plushnet,testdata)
c = categories(trainlabels)
testpred = scores2label(testscores,c)
confusionchart(testlabels,testpred)

%%
%[text] 
%[text] Try bidirectional layer and retrain:
layers = [
    sequenceInputLayer(11)
    bilstmLayer(512,OutputMode="last")
    dropoutLayer(0.5)
    fullyConnectedLayer(18)
    softmaxLayer()];

opts = trainingOptions("adam", ...
    ValidationData={valdata vallabels}, ...
    Plots="training-progress", ...
    Metrics="accuracy", ...
    MaxEpochs=150, ...
    MiniBatchSize=numel(trainlabels), ...
    ValidationFrequency=10);

plushnet = trainnet(traindata,trainlabels,layers,"crossentropy",opts)
%%
%[text] Retest accuracy of the improved model
acc = testnet(plushnet,testdata,testlabels,"accuracy")
testscores = minibatchpredict(plushnet,testdata);
testpred = scores2label(testscores,c);
confusionchart(testlabels,testpred)
%%
%[text] Test another architecture - CNN
convoLayers = [
    sequenceInputLayer(11,"Name","input")
    convolution1dLayer(28,32,"Name","convolution1","Padding","causal","PaddingValue","symmetric-exclude-edge") % return causal?  change filter size = looking back into history
    reluLayer("Name","relu1")
    layerNormalizationLayer("Name","layernorm1")
    convolution1dLayer(32,64,"Name","convolution2","Padding","causal","PaddingValue","symmetric-exclude-edge","DilationFactor",2) % return causal?
    reluLayer("Name","relu2")
    layerNormalizationLayer("Name","layernorm2")
    convolution1dLayer(41,64,"Name","convolution3","Padding","causal","PaddingValue","symmetric-exclude-edge","DilationFactor",4) % return causal?
    reluLayer("Name","relu3")
    layerNormalizationLayer("Name","layernorm3")
    globalAveragePooling1dLayer("Name","averagepool")
    fullyConnectedLayer(18,"Name","fc")
    softmaxLayer("Name","softmax")];

convoOpts = trainingOptions("adam", ...
    ValidationData={valdata vallabels}, ...
    Plots="training-progress", ...
    Metrics="accuracy", ...
    MaxEpochs=450, ...
    MiniBatchSize=numel(trainlabels), ...
    ValidationFrequency=10);

convoPlushNet = trainnet(traindata,trainlabels,convoLayers,"crossentropy",convoOpts)

%%
%[text] Retest accuracy of the convolution model
acc = testnet(convoPlushNet,testdata,testlabels,"accuracy")
testscores = minibatchpredict(convoPlushNet,testdata);
testpred = scores2label(testscores,c);
confusionchart(testlabels,testpred)

%%
%[text] Save the model for later testing on real-time data, if you are satisfied with the results
% In your training session (once)
classNames = [ ...
"At rest","Back stroke 1","Back stroke 2","Ear scratch 1","Ear scratch 2", ...
"Head stroke 1","Head stroke 2","Hold in hands","Hold strong","Neck scratch 1", ...
"Neck scratch 2","Tail hit 1","Tail hit 2","Tail pull","Tail scratch 1", ...
"Tail scratch 2","Tail tap 1","Tail tap 2"];

save('convoPlushNet-NewData5k-d1d2d4.mat','convoPlushNet','classNames');  % just the net + label order

%%
%[text] Tedt dilated 1-D CNN for MCU
% Dilated causal CNN for MCU use (softmax last; trainable with trainnet)
% 11 inputs @100 Hz -> 18 classes
convoLayersMCU = [
    sequenceInputLayer(11,"Name","input")

    % Layer 1 – small kernel, dilation 1
    convolution1dLayer(14,6,"Name","conv1", ...
        "Padding","causal","DilationFactor",1,"PaddingValue","symmetric-exclude-edge")
    reluLayer("Name","relu1")

    % Layer 2 – wider kernel, dilation 2
    convolution1dLayer(39,8,"Name","conv2", ...
        "Padding","causal","DilationFactor",2,"PaddingValue","symmetric-exclude-edge")
    reluLayer("Name","relu2")

    % Layer 3 – widest kernel, dilation 4
    convolution1dLayer(41,10,"Name","conv3", ...
        "Padding","causal","DilationFactor",4,"PaddingValue","symmetric-exclude-edge")
    reluLayer("Name","relu3")

    globalAveragePooling1dLayer("Name","gap")
    fullyConnectedLayer(18,"Name","fc")
    softmaxLayer("Name","softmax")
];



%%
%[text] Train the MCU network
convoOpts = trainingOptions("adam", ...
    ValidationData={valdata vallabels}, ...
    Plots="training-progress", ...
    Metrics="accuracy", ...
    MaxEpochs=2000, ...
    MiniBatchSize=numel(trainlabels), ...
    ValidationFrequency=10);

convoPlushNetMCU = trainnet(traindata,trainlabels,convoLayersMCU,"crossentropy",convoOpts)


%%
%[text] Retest accuracy of the convolution model
acc = testnet(convoPlushNetMCU,testdata,testlabels,"accuracy")
testscores = minibatchpredict(convoPlushNetMCU,testdata);
c = categories(trainlabels);
testpred = scores2label(testscores,c);
confusionchart(testlabels,testpred)

%[text]  
%%
%[text] Save the optimized MCU model for real-time testing
% In your training session (once)
classNames = [ ...
"At rest","Back stroke 1","Back stroke 2","Ear scratch 1","Ear scratch 2", ...
"Head stroke 1","Head stroke 2","Hold in hands","Hold strong","Neck scratch 1", ...
"Neck scratch 2","Tail hit 1","Tail hit 2","Tail pull","Tail scratch 1", ...
"Tail scratch 2","Tail tap 1","Tail tap 2"];

save('convoPlushNetMCU.mat','convoPlushNetMCU','classNames');  % just the net + label order


%[appendix]{"version":"1.0"}
%---
%[metadata:view]
%   data: {"layout":"onright","rightPanelPercent":30}
%---
