function [net, testResults] = train_and_test_lstm(params, master)

if nargin < 2
    master = load(params.masterFile);
    master = master.master;
end

% -------------------------------------------------------------------------

trainTargets = params.trainTargets;
testTargets = params.testTargets;
lstmPath = fullfile(params.outFolder, "LSTM.mat");

% -------------------------------------------------------------------------

if ismember("LSTM",params.retrain)
    
    if exist(lstmPath)
        inp = input(sprintf("%s already exists, want to overwrite [y/n]?:",lstmPath), ...
                    's');
        if ~strcmp(inp,"y")
            fprintf("Aborting training of LSTM\n\n")
            overallMetrics = []; perRecMetrics = []; net=[];        
            
        end
    end
    
    layers = init_lstm(params);
    if gpuDeviceCount()
        execEnv = "gpu";
    else
        execEnv = "cpu";
    end
    trainingSettings = trainingOptions("adam", ...
                                       "InitialLearnRate", 1e-4, ...
                                        "Plots","training-progress",...
                                        "MaxEpochs", 16 , ...
                                        "ExecutionEnvironment", execEnv, ...
                                         "Shuffle","every-epoch");

    % Generate training data
    trainData = master(master.train,:);
    
    [trainIdx, ~] = findgroups(trainData.id);
    xTrainLSTM = splitapply(@(x,id) {create_lstm_data(x,id,params)}, trainData(:,["features","id"]), trainIdx);
    xTrainLSTM = vertcat(xTrainLSTM{:});
    
    yTrain = trainData{:,trainTargets};
    validTrainingIndex = ~isnan(yTrain);  
    yTrainLSTM = categorical(yTrain(validTrainingIndex));
    
    xTrainLSTM = xTrainLSTM(validTrainingIndex);

    trainedLSTM = trainNetwork(xTrainLSTM, yTrainLSTM, layers, trainingSettings);
    save(lstmPath,"trainedLSTM");
end

trainedLSTM = load(lstmPath,'-mat');
trainedLSTM = trainedLSTM.trainedLSTM;

% Create LSTM test data
testData = master(~master.train,:);

[testIdx, ~] = findgroups(testData.id);
xTestLSTM = splitapply(@(x,id) {create_lstm_data(x, id, params)}, testData(:,["features","id"]), testIdx);
xTestLSTM = vertcat(xTestLSTM{:});

yTestLSTM = testData{:,testTargets};

testTable = table;
testTable.id = testData.id;
testTable.X = xTestLSTM;
testTable.Y = yTestLSTM;

% Predict on each recording
[idx, id] = findgroups(testTable.id);

perRecPredictions = splitapply(@(x) classify_and_post_process(x, trainedLSTM, params), ...
                              testTable.X, idx);

perRecTargets = splitapply(@(x) num2cell(x,1), testTable.Y, idx);


perRecProbs = splitapply(@(x){ trainedLSTM.predict(x) }, testTable.X, idx);
perRecProbs = cellfun(@(x) double(x(:,2)), perRecProbs, "UniformOutput",false);

testResults = table;
testResults.id = id;
testResults.yTrue = perRecTargets;
testResults.yHat  = perRecPredictions;
testResults.probs = perRecProbs;

% perRecMetrics = cellfun(@(x,y) calc_performance_metrics(x,y,params), ...
%                         perRecTargets, perRecPredictions);
% 
% perRecMetrics = struct2table(perRecMetrics);
% perRecMetrics.id = id;
% 
% % Predict on concatenated data
% 
% overallMetrics = calc_performance_metrics(...
%                                           vertcat(perRecTargets{:}), ...
%                                           vertcat(perRecPredictions{:}), ...
%                                           params ...
%                                           );

net = trainedLSTM;

end

function layers = init_lstm(params)
    
layers = [...
    
    sequenceInputLayer(params.numFeatures, "Name","sequenceinput"),
    
    lstmLayer(100, "Name", "lstm_1", ...
              "InputWeightsInitializer","narrow-normal",...
              "RecurrentWeightsInitializer","narrow-normal" ...
              ), ...
    
    dropoutLayer(0.3, "Name","dropout_1"), ...
    
    lstmLayer(100, "Name", "lstm_2", ...
              "InputWeightsInitializer","narrow-normal",...
              "RecurrentWeightsInitializer","narrow-normal", ...
              "OutputMode","last"), ...

    dropoutLayer(0.3, "Name","dropout_2"), ...

    fullyConnectedLayer(2, "Name", "fc", ...
                        "BiasInitializer","zeros", ...
                        "WeightsInitializer","narrow-normal"), ...
    
    softmaxLayer("Name","softmax"), ...

    classificationLayer("Name", "classoutput")
];

end


function data = create_lstm_data(features, name, params)

if nargin < 2; name = ""; end
fprintf("Creating LSTM data for %s\n",unique(name));

numFeatures = params.numFeatures;
winSizeSamples = params.windowSizeLSTM;
numWins = height(features) - (winSizeSamples-1);
winMask = repmat(1:winSizeSamples,numWins,1);
winStep = 1 * (0:numWins-1)';
windows = winMask+winStep;

middleSeqX = cellfun(@(x) mat2cell(vertcat(features{x})', numFeatures, size(x,2)), ...
                    num2cell(windows,2));

startPadIdx = generate_padding_indexes(winSizeSamples);
endPadIdx = (height(features)+1)-flip(startPadIdx,2); 
endPadIdx(endPadIdx>height(features)) = 0;

startPadSeqX = cellfun(@(x) mat2cell(extract_and_pad(features, x, "left"), ...
                                     numFeatures, size(x,2)), ...
                       num2cell(startPadIdx,2));

endPadSeqX = cellfun(@(x) mat2cell(extract_and_pad(features, x, "right"), ...
                                   numFeatures, size(x,2)), ...
                    num2cell(endPadIdx,2));

% Concat data
data = [startPadSeqX;
        middleSeqX;
        endPadSeqX];

end


function out = generate_padding_indexes(winSizeSamples)

    outsideWindow = (winSizeSamples-1)/2;
    samples = 1:winSizeSamples;
    out = nan(outsideWindow,winSizeSamples);
    for i = 1:outsideWindow
        numPads = (outsideWindow+1)-i;
        numPrev = i - 1;
        if numPrev; idxPrev = 1:numPrev; else; idxPrev = []; end
        idxNext = i:i+outsideWindow;
        out(i,:)=[zeros(1,numPads),samples(idxPrev),samples(idxNext)];
    end

end


function out = extract_and_pad(in, idx, side)
    pad_idx = idx==0;
    numFeatures = size(in{1},2);
    z = zeros(numFeatures,sum(pad_idx));
    x = vertcat(in{idx(~pad_idx)})';
    if side == "left"; out = [z, x]; 
    elseif side == "right"; out = [x, z]; end
end

function [processedPredictions] = classify_and_post_process(X, model, params)


yHatCategorical = categorical(classify(model, X));
yHatRaw = grp2idx(yHatCategorical)-1;
yHatRaw(yHatRaw==0) = -1;

% Apply post-processing

% 1. Remove MS predictions shorter than 1 seconds
yHat = remove_invalid_labels(yHatRaw, params.minDurationSec, params.maxDurationSec, ...
                             params.windowSizeSec, params.replaceInvalidPredictions, ...
                             params.skipSingles);
    
% 2. Apply 9s moving median filter to RFmodel and SVMmodel predictions
% yHat = movmedian(fixedYhat,45,"Endpoints","shrink");
% yHat(yHat==0) = 1;

processedPredictions = num2cell(yHat,1);

end
