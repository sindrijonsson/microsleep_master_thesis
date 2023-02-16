
%% LSTM net

layers = [...
    
    sequenceInputLayer(14, "Name","sequenceinput"),
    
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

%% LSTM Training options
trainingSettings = trainingOptions("adam", ...
    "Plots","training-progress",...
    "MaxEpochs",16, ...
    "MiniBatchSize",256, ...
    "CheckpointPath",".\checkpoint",...
    "Shuffle","every-epoch");


%% Generate LSTM training data (with padding for non overlapping labels)

master = load("master.mat");
data = master.checkPoint.dataTable;
trainData = data(data.train,:);
testData = data(~data.train,:);

[trainIdx, ~] = findgroups(trainData.patient);
xTrainLSTM = splitapply(@(x,id) {create_lstm_data(x,id)}, trainData(:,["features","patient"]), trainIdx);
xTrainLSTM = vertcat(xTrainLSTM{:});

trainTargets = trainData.biWake_vs_biMSE;
validTrainingIndex = ~isnan(trainTargets);  
yTrainLSTM = categorical(trainTargets(validTrainingIndex));

xTrainLSTM = xTrainLSTM(validTrainingIndex);

%% Train the network
% lstmNet = trainNetwork(xTrainLSTM,yTrainLSTM,layers,trainingSettings)

%% Load the trained network
trainedLSTM = load("checkpoint\trained_LSTM.mat","-mat");

%% Create LSTM test data

[testIdx, pat] = findgroups(testData.patient);
xTestLSTM = splitapply(@(x,id) {create_lstm_data(x,id)}, testData(:,["features","patient"]), testIdx);
xTestLSTM = vertcat(xTestLSTM{:});

testTargets = testData.biWake_vs_biMSE;
% validTrainingIndex = ~isnan(targets);  
% yTrainLSTM = categorical(targets(validTrainingIndex));
yTestLSTM = testTargets;

% xTrainLSTM = xTrainLSTM(validTrainingIndex);
testTable = table;
testTable.rec = testData.patient;
testTable.X = xTestLSTM;
testTable.Y = yTestLSTM;


%% Predict on each recording
[idx, rec] = findgroups(testTable.rec);

perRecPredictions = splitapply(@(x) classify_and_post_process(x,trainedLSTM.net,1), ...
                              testTable.X, idx);

perRecTargets = splitapply(@(x) num2cell(x,1), testTable.Y, idx);
%%
perRecPerformance = cellfun(@(x,y) calc_performance_metrics(x,y,[1,-1]), ...
                            perRecTargets, perRecPredictions);

perRecOrgLabels = arrayfun(@(x)get_labels(sprintf("data/%s.mat",x),1),rec);

overallPerformance = calc_performance_metrics(...
                                              vertcat(perRecTargets{:}), ...
                                              vertcat(perRecPredictions{:}) ...
                                              );



