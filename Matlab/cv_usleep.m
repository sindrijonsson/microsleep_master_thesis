function [cvResults, optHz, optMethod] = cv_usleep(params, usleep, master)

seed = params.seed;
trainTargets = params.trainTargets;
testTargets = params.testTargets;

thresholds = params.usleep.thresholds;
HZ = params.usleep.hz;
evalMetric = params.usleep.evalMetric;

% -------------------------------------------------------------------------
% Create CV partition

df = table;
[idx, rec] = findgroups(master.id);
df.rec = rec;
df.train = splitapply(@(x) all(x), master.train, idx);
df.MS = splitapply(@(x) any(x==1), master(:,trainTargets), idx);

devDf = df(df.train,:);
testDf = df(~df.train,:);

rng(seed.generator);
cv = cvpartition(devDf.MS, 'KFold', 5, 'Stratify', true);


% -------------------------------------------------------------------------


cvResults = table;

for hIdx = 1:numel(HZ)
    
    hz = HZ(hIdx);
    
    tmpTable = usleep(usleep.hz == hz,:);
    methods = tmpTable.Properties.VariableNames((string(tmpTable.Properties.VariableNames).startsWith("preds")));
    
    for mIdx = 1:numel(methods)

    
        method = string(methods(mIdx));


        if any(cellfun(@(x) isempty(x), tmpTable{:,method}))
            fprintf("Skipping %s @ %i Hz\n",method,hz)
            continue
        end

        for k = 1:cv.NumTestSets

            trainIdx = cv.training(k);
            testIdx = cv.test(k);
            
            trainIdx = ismember(tmpTable.id, devDf.rec(trainIdx));
            testIdx = ismember(tmpTable.id, devDf.rec(testIdx));
            
            [kMetrics, optIdx] = k_eval_usleep(trainIdx, testIdx, tmpTable, method, params);

            if contains(method, "Argmax")
                kMetrics.optIdx = nan;
                kMetrics.optThres = nan;
            else
                kMetrics.optIdx = optIdx;
                kMetrics.optThres = thresholds(optIdx);
            end
            
            fprintf("Cross-validating %s @ %i Hz (k=%i):" + ...
                " %s = %.2f Threshold = %.3f \n", ...
                method,hz,k, ...
                evalMetric, kMetrics.(evalMetric), kMetrics.optThres);
            
            kMetrics.k = k;
            kMetrics.method = method;
            kMetrics.hz = hz;
            
            cvResults = [cvResults; struct2table(kMetrics)];
            
        end
    end

end


%% Find the best method
[idx, hz, method] = findgroups(cvResults.hz, cvResults.method);
cvEvalMean = splitapply(@(x) mean(x,"omitnan"), cvResults{:,evalMetric}, idx);
cvEvalSem = splitapply(@(x) std(x,"omitnan")/sqrt(height(x)), cvResults{:,evalMetric}, idx);
cvModels = table;
cvModels.hz = hz;
cvModels.method = method;
cvModels.eval = cvEvalMean;
cvModels.evalSem = cvEvalSem;

[optEval, optIdx] = max(cvModels.eval);
optMethod = cvModels.method(optIdx);
optHz = cvModels.hz(optIdx);
fprintf("The best model is %s @ %i Hz (%s = %.2f)\n", ...
    optMethod, optHz, evalMetric, optEval)

cvFile = fullfile(params.outFolder, "cv_usleep.mat");
cv = struct;
cv.cvResults = cvResults;
cv.optHz = optHz;
cv.optMethod = optMethod;
save(cvFile,"cv");

end

% 
% %% Retrain the best model on the entire training set to find optimal threshold
% trainIdx = ismember(bestModel.name, devDf.patient);
% 
% trainPred = bestModel{trainIdx,bestMethod};
% trainPred = num2cell(horzcat(trainPred{:}),2);
% 
% trainTarget = bestModel{trainIdx,"target"};
% trainTarget = horzcat(trainTarget{:});
% 
% kTrainMetrics = cellfun(@(x) calc_performance_metrics(trainTarget, x), trainPred);
% 
% % Find optimal threshold
% [~, optIdx] = max(vertcat(kTrainMetrics.kappa));
% 
% %% 
% testIdx = ismember(bestModel.name, testDf.patient);
% 
% perRecMetrics = cellfun(@(x,y) calc_performance_metrics(x,y), ...
%                         bestModel{testIdx,"target"}, ...
%                         bestModel{testIdx,bestMethod});
% 
% nanmean(table2array(struct2table(perRecMetrics)))
% 
% %%
% 
% testPred = bestModel{testIdx,bestMethod};
% testPred = num2cell(horzcat(testPred{:}),2);
% testPred = testPred{optIdx,:};
% 
% testTarget = bestModel{testIdx,"target"};
% testTarget = horzcat(testTarget{:});
% 
% testMetrics = calc_performance_metrics(testTarget, testPred)
% 
% end
% 
%%

function [kTestMetrics, optIdx] = k_eval_usleep(trainIdx, testIdx, dataTable, method, params)

trainPred = dataTable{trainIdx,method};
trainPred = num2cell(horzcat(trainPred{:}),2);

trainTarget = dataTable{trainIdx,"trainTargets"};
trainTarget = horzcat(trainTarget{:});

kTrainMetrics = cellfun(@(x) calc_performance_metrics(trainTarget, x, params), trainPred);

% Find optimal threshold
[~, optIdx] = max(vertcat(kTrainMetrics.(params.usleep.evalMetric)));

testPred = dataTable{testIdx,method};
testPred = horzcat(testPred{:});
testPred = testPred(optIdx,:);

testTarget = dataTable{testIdx,"trainTargets"};
testTarget = horzcat(testTarget{:});

kTestMetrics = calc_performance_metrics(testTarget, testPred);

end

% end