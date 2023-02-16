function [optModel, testResults, optThres] = train_and_test_usleep(usleep, optHz, optMethod, params)

trainTargets = params.trainTargets;
testTargets = params.testTargets;
split = params.split;

thresholds = params.usleep.thresholds;
HZ = params.usleep.hz;
evalMetric = params.usleep.evalMetric;

% -------------------------------------------------------------------------

trainIds = usleep.id(usleep.train);
testIds = usleep.id(~usleep.train);

% -------------------------------------------------------------------------


% Retrain the best model on the entire training set to find optimal thresholds

optModel = usleep(usleep.hz == optHz,:); 

trainIdx = ismember(optModel.id, trainIds);

trainPred = optModel{trainIdx,optMethod};
trainPred = num2cell(horzcat(trainPred{:}),2);

trainTarget = optModel{trainIdx,"trainTargets"};
trainTarget = horzcat(trainTarget{:});

kTrainMetrics = cellfun(@(x) calc_performance_metrics(trainTarget, x), trainPred);

% Find optimal threshold
[~, optIdx] = max(vertcat(kTrainMetrics.(evalMetric)));
optThres = thresholds(optIdx);

% -------------------------------------------------------------------------
% Evaluate

testIdx = ismember(optModel.id, testIds);

testIds = optModel.id(testIdx);
perRecPreds = optModel{testIdx, optMethod};
perRecPreds = cellfun(@(x) x(optIdx,:), perRecPreds, 'UniformOutput', false);
perRecTargets = optModel{testIdx,"testTargets"};
perRecProbs = optModel{testIdx, "probs"};

if contains(optMethod,"max")
    perRecProbs = cellfun(@(x) max(x(2:end,:),[],1), perRecProbs, 'UniformOutput',false);
elseif contains(optMethod, "sum")
    perRecProbs = cellfun(@(x) sum(x(2:end,:),[],1), perRecProbs, 'UniformOutput',false);
else
    perRecProbs = perRecProbs;
end

testResults = table;
testResults.id = testIds;
testResults.yTrue = perRecTargets;
testResults.yHat = perRecPreds;
testResults.probs = perRecProbs;

% perRecMetrics = cellfun(@(x,y) calc_performance_metrics(x,y,params), ...
%                         perRecTargets, ...
%                         perRecPreds);
% perRecMetrics = struct2table(perRecMetrics);
% perRecMetrics.id = optModel{testIdx,"id"};
% 
% testPreds = horzcat(perRecPreds{:});
% testTarget = horzcat(perRecTargets{:});
% 
% overallMetrics = calc_performance_metrics(testTarget, testPreds, params);

% -------------------------------------------------------------------------
% Save model

optPredictions = cellfun(@(x) x(optIdx,:), optModel{:,optMethod}, 'UniformOutput',false);

model = optModel(:,["id","train","trainTargets","testTargets"]);
model.optPredictions = optPredictions;
save(params.usleep.outfile, "model");


end

