function [model, testResults] = train_and_test_model(modelName, params, master)

if nargin < 2
    master = load(params.masterFile);
    master = master.master;
end

% -------------------------------------------------------------------------

trainTargets = params.trainTargets;
testTarget = params.testTargets;

negClassValue = params.negClassValue;
posClassValue = params.posClassValue;

seed = params.seed;
retrainModel = ismember(modelName,params.retrain);
modelFile = fullfile(params.outFolder,sprintf("%s.mat",modelName));

% -------------------------------------------------------------------------
% Collect the training data

trainData = master(master.train,:);

yTrain = trainData{:,trainTargets};

% Collect negative and positive classes
posClasses = find(yTrain==posClassValue);
negClasses = find(yTrain==negClassValue);

% Select random negative classes (balance with positive classes)
rng(seed.generator);
negIdx = randi(length(negClasses),[length(posClasses),1]);
useNegClasses = negClasses(negIdx);

useIdx = [posClasses; useNegClasses];
trainX = vertcat(trainData.features{useIdx});
trainY = categorical(yTrain(useIdx));

% -------------------------------------------------------------------------
% Train the specified model (either RF or SVM) using the training data


if retrainModel
fprintf("Training %s model... \n", modelName)

    if exist(modelFile)
        inp = input(sprintf("%s already exists, want to retrain [y/n]?:",modelFile), ...
                    's');
    
        if ~strcmp(inp,"y")
            fprintf("Aborting training of %s\n\n",modelName)
            overallMetrics = []; perRecMetrics = []; model=[];
            
        end
    end
    switch modelName
    
        case "RF"
    
            model = TreeBagger(100, trainX, trainY, ...
                    "Method", "classification", ...
                    "SampleWithReplacement", "on", ...
                    "InBagFraction", 1, ...
                    "OOBPrediction","on", ...
                    "NumPredictorsToSample", 4, ...
                    "MinLeafSize", 1);
    
        case "SVM"
    
            model = fitcsvm(trainX, trainY, ...
                "KernelFunction", "gaussian", ...
                "BoxConstraint", 1, ...
                "KernelScale", "auto",...
                "Solver", "SMO", ...
                "Standardize",false, ...
                "Verbose",1);
    
    end

else
    fprintf("Loading %s model from %s... \n", modelName, modelFile)
    load(modelFile,"model");
end

% -------------------------------------------------------------------------
fprintf("Evaluating %s model... \n", modelName)

% Evaluate model
testData = master(~master.train,:);
testTable = table;
testTable.id = testData.id;
testTable.X = vertcat(testData.features{:});
testTable.Y = vertcat(testData{:,testTarget});
    
% Make predictions per recording with post-processing
[idx, id] = findgroups(testTable.id);

perRecPredictions = splitapply(@(x) predict_and_post_process(x,model,params), ...
                              testTable.X, idx);

perRecTargets = splitapply(@(x) num2cell(x,1), testTable.Y, idx);

testResults = table;
testResults.id = id;
testResults.yTrue = perRecTargets;
testResults.yHat  = perRecPredictions;


% perRecMetrics = cellfun(@(x,y) calc_performance_metrics(x,y,params), ...
%                           perRecTargets, perRecPredictions);
% 
% perRecMetrics = struct2table(perRecMetrics);
% perRecMetrics.id = id;
% 
% % perRecOrgLabels = arrayfun(@(x)get_labels(sprintf("data/%s.mat",x),0),rec);
% 
% overallMetrics = calc_performance_metrics( ...
%                                           vertcat(perRecTargets{:}), ...
%                                           vertcat(perRecPredictions{:}), ...
%                                           params ...
%                                           );

% -------------------------------------------------------------------------


% Save the model
if ismember(modelName,params.retrain)
    save(modelFile,"model");
end

end

% -------------------------------------------------------------------------

function [processedPredictions] = predict_and_post_process(X, model, params)

yHatCategorical = categorical(predict(model, X));
yHatRaw = grp2idx(yHatCategorical)-1;
yHatRaw(yHatRaw==0) = params.negClassValue;

% Apply post-processing


% 1. Apply 9s moving median filter to RFmodel and SVMmodel predictions
yHat = movmedian(yHatRaw, params.windowSizeLSTM, "Endpoints", "shrink");
yHat(yHat==0) = params.posClassValue;

% 2. Remove MS predictions shorter than 1 seconds
yHat = remove_invalid_labels(yHat, params.minDurationSec, params.maxDurationSec, ...
                                  params.windowSizeSec, params.replaceInvalidPredictions, ...
                                  params.skipSingles);

processedPredictions = num2cell(yHat,1);


end

% function [processedPredictions] = predict_and_post_process(X, model, params)
% 
% yHatCategorical = categorical(predict(model, X));
% yHatRaw = grp2idx(yHatCategorical)-1;
% yHatRaw(yHatRaw==0) = -1;
% 
% % Apply post-processing
% 
% % 1. Remove MS predictions shorter than 1 seconds
% fixedYhat = remove_invalid_labels(yHatRaw, params.minDurationSec, params.maxDurationSec, ...
%                                   params.windowSizeSec, params.replaceInvalidPredictions);
% 
% % 2. Apply 9s moving median filter to RFmodel and SVMmodel predictions
% yHat = movmedian(fixedYhat, params.windowSizeLSTM, "Endpoints", "shrink");
% yHat(yHat==0) = 1;
% 
% processedPredictions = num2cell(yHat,1);
% 
% 
% end