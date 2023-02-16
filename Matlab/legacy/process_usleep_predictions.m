close all
clear all
clc
%%

seed = load("seed.mat");

HZ = [128, 64, 32, 16, 8, 1];
names=unique(string(ls('predictions\*\*.mat')));
mz = load("master_zeropad.mat");

% thresholds
thresholds = (0.025:0.025:1)';

%% Process u-sleep probabilites and save

for hIdx = 1:numel(HZ)
    hz = HZ(hIdx);
    if hz==1; break; end
    hzTable = table;
    for nIdx = 1:numel(names)

        name = names(nIdx);

        p = sprintf("predictions\\%i_hz\\%s",hz,name);
        fprintf("Processing %s\n",p);
        tmp=load(p);
        tmp = double(tmp.data);
        tmpName = p.extractAfter("hz\").extractBefore(".mat");

        % Get reference data from master
        data=mz.checkPoint.dataTable;
        d=data(data.patient == tmpName,:);

        % Get indexs for resampling the predictions to match the targets
        resampIdxs = convert_target_time_to_prediction_samples(d.sample, 0.2, hz);

        % convert cumulative probabilties to 0 - 1
        probs = tmp ./ sum(tmp,2);
        
        % preds argmax
        [~, aasmPredsArgmax] = max(probs,[],2);
        predsArgmax = aasm_to_wake_vs_sleep(aasmPredsArgmax)';
        predsArgmax = iter_resample(predsArgmax, resampIdxs, 0);

        % probs sum
        probsSum = sum(probs(:,2:end),2)';
        predsSum = probsSum > thresholds;
        predsSum = iter_resample(predsSum, resampIdxs, 0.5);
        predsSum = iter_remove_invalid_labels(predsSum);
        predsSum(predsSum==0) = -1;

        % probs max
        probsMax = max(probs(:,2:end),[],2)';
        predsMax = probsMax > thresholds;
        predsMax = iter_resample(predsMax, resampIdxs, 0.5);
        predsMax = iter_remove_invalid_labels(predsMax);
        predsMax(predsMax==0) = -1;

        % Using the mean within bin for sum of sleep probabilites
        probsBinSum = iter_bin_calc(probsSum, resampIdxs, @(x) mean(x,2));
        predsBinSum = probsBinSum > thresholds;
        predsBinSum = iter_remove_invalid_labels(predsBinSum);
        predsBinSum(predsBinSum==0) = -1;

        % Using the mean within bin for the max sleep probability
        probsBinMax = iter_bin_calc(probsMax, resampIdxs, @(x) mean(x,2));
        predsBinMax = probsBinMax > thresholds;
        predsBinMax = iter_remove_invalid_labels(predsBinMax);
        predsBinMax(predsBinMax==0) = -1;
    
        
        % Collect into table
        entry = struct;
        compress = @(x) mat2cell(x,size(x,1),size(x,2));
        entry.name          = tmpName;
        entry.hz            = hz;
        entry.target        = compress(d.biWake_vs_biMSE');
        entry.predsArgmax   = compress(predsArgmax);
        entry.predsSum      = compress(predsSum);
        entry.predsMax      = compress(predsMax);
        entry.predsBinSum   = compress(predsBinSum);
        entry.predsBinMax   = compress(predsBinMax);

        % Append
        hzTable = [hzTable; struct2table(entry)];

    end
    save(sprintf("%i_processed_usleep_predictions.mat",hz),"hzTable");
end

%% A bit different for 1 hz
hzTable = table;
for nIdx = 1:numel(names)

        name = names(nIdx);

        p = sprintf("predictions\\1_hz\\%s",name);
        fprintf("Processing %s\n",p);
        tmp=load(p);
        tmp = double(tmp.data);
        tmpName = p.extractAfter("hz\").extractBefore(".mat");

        % Get reference data from master
        data=mz.checkPoint.dataTable;
        d=data(data.patient == tmpName,:);

        % convert cumulative probabilties to 0 - 1
        probs = tmp ./ sum(tmp,2);

        % preds argmax
        [~, aasmPredsArgmax] = max(probs,[],2);
        predsArgmax = aasm_to_wake_vs_sleep(aasmPredsArgmax)';
        predsArgmax = repelem(double(predsArgmax), 1, 5);

        % probs sum
        probsSum = sum(probs(:,2:end),2)';
        predsSum = double(probsSum > thresholds);
        predsSum = repelem(predsSum, 1, 5);
        predsSum(predsSum==0) = -1;

        % probs max
        probsMax = max(probs(:,2:end),[],2)';
        predsMax = double(probsMax > thresholds);
        predsMax = repelem(predsMax, 1, 5);
        predsMax(predsMax==0) = -1;

        % Collect into table
        entry = struct;
        compress = @(x) mat2cell(x,size(x,1),size(x,2));
        entry.name          = tmpName;
        entry.target        = compress(d.biWake_vs_biMSE');
        entry.hz            = hz;
        entry.predsArgmax   = compress(predsArgmax);
        entry.predsSum      = compress(predsSum);
        entry.predsMax      = compress(predsMax);

        % Append
        hzTable = [hzTable; struct2table(entry)];
end
% save("1_processed_usleep_predictions.mat","hzTable");

%% 5-fold crossvalidation

% Partition data into k-fold using training split 
% and stratify by presence of MS

df = table;
[idx, pat] = findgroups(data.patient);
df.patient = pat;
df.train = splitapply(@(x) all(x), data.train, idx);
df.MS = splitapply(@(x) any(x==1), data.biWake_vs_biMSE, idx);

devDf = df(df.train,:);
testDf = df(~df.train,:);

rng(seed.generator);
cv = cvpartition(devDf.MS, 'KFold', 5, 'Stratify', true);

cvResults = table;

for hIdx = 1:numel(HZ)
    
    hz = HZ(hIdx);
    tmpTable = load(sprintf("%i_processed_usleep_predictions",hz));
    hzTable = tmpTable.hzTable;
    methods = hzTable.Properties.VariableNames((string(hzTable.Properties.VariableNames).startsWith("preds")));
    
    for mIdx = 1:numel(methods)
    
        method = string(methods(mIdx));

        for k = 1:cv.NumTestSets

            fprintf("Processing %s @ %i Hz (k=%i)\n",method,hz,k)
            trainIdx = cv.training(k);
            testIdx = cv.test(k);
            
            trainIdx = ismember(hzTable.name, devDf.patient(trainIdx));
            testIdx = ismember(hzTable.name, devDf.patient(testIdx));
            
            [kMetrics, optIdx] = k_eval_usleep(trainIdx, testIdx, hzTable, method);

            if method == "predsArgmax"
                kMetrics.optIdx = nan;
                kMetrics.optThres = nan;
            else
                kMetrics.optIdx = optIdx;
                kMetrics.optThres = thresholds(optIdx);
            end

            kMetrics.k = k;
            kMetrics.method = method;
            kMetrics.hz = hz;
            
            cvResults = [cvResults; struct2table(kMetrics)];
            
        end
    
    end

end

%% Find the best method
[idx, hz, method] = findgroups(cvResults.hz, cvResults.method);
cvKappaMean = splitapply(@(x) mean(x), cvResults.kappa, idx);
cvKappaSem = splitapply(@(x) std(x)/sqrt(height(x)), cvResults.kappa, idx);
cvModels = table;
cvModels.hz = hz;
cvModels.method = method;
cvModels.kappa = cvKappaMean;
cvModels.kappaSem = cvKappaSem;

[bestKappa, bestIdx] = max(cvModels.kappa);
bestMethod = cvModels.method(bestIdx);
bestHz = cvModels.hz(bestIdx);
fprintf("The best model is %s @ %i Hz (kappa = %.2f)\n", ...
    bestMethod, bestHz, bestKappa)

bestModel = load(sprintf("%i_processed_usleep_predictions.mat",bestHz));
bestModel = bestModel.hzTable;
%% Retrain the best model on the entire training set to find optimal threshold
trainIdx = ismember(bestModel.name, devDf.patient);

trainPred = bestModel{trainIdx,bestMethod};
trainPred = num2cell(horzcat(trainPred{:}),2);

trainTarget = bestModel{trainIdx,"target"};
trainTarget = horzcat(trainTarget{:});

kTrainMetrics = cellfun(@(x) calc_performance_metrics(trainTarget, x), trainPred);

% Find optimal threshold
[~, optIdx] = max(vertcat(kTrainMetrics.kappa));

%% 
testIdx = ismember(bestModel.name, testDf.patient);

perRecMetrics = cellfun(@(x,y) calc_performance_metrics(x,y), ...
                        bestModel{testIdx,"target"}, ...
                        bestModel{testIdx,bestMethod});

nanmean(table2array(struct2table(perRecMetrics)))

%%

testPred = bestModel{testIdx,bestMethod};
testPred = num2cell(horzcat(testPred{:}),2);
testPred = testPred{optIdx,:};

testTarget = bestModel{testIdx,"target"};
testTarget = horzcat(testTarget{:});

testMetrics = calc_performance_metrics(testTarget, testPred)

%%

function out = aasm_to_wake_vs_sleep(probs)
out=probs;
out(probs==1) = -1;
out(probs~=1) = 1;
end

function samples = convert_target_time_to_prediction_samples(samples, step, prediction_hz)

t1 = [0; samples(1:end-1)*step];
t2 = samples*step;
maxPredictionSamples = (length(samples) * step) * prediction_hz;
samples=[max(1,floor(t1*prediction_hz)), min(ceil(t2*prediction_hz),maxPredictionSamples)];

end

function out = iter_resample(y, reIdx, splitVal)
out = nan(size(y,1),size(reIdx,2));
for i = 1:height(reIdx)
    out(:,i) = median(y(:,reIdx(i,1):reIdx(i,2)), 2);
end
out(out==splitVal) = 1;
end


function out = iter_bin_calc(y, binIdx, fcn)
out = nan(size(y,1),size(binIdx,2));
for i = 1:height(binIdx)
    bin = y(:,binIdx(i,1):binIdx(i,2));
    out(:,i) = fcn(bin);
end
end

function out = iter_remove_invalid_labels(y, params)
out = zeros(size(y));
for i = 1:height(y)
    out(i,:) = remove_invalid_labels(y(i,:), params.minDurationSec, params.maxDurationSec, ...
                                    params.windowSizeSec, params.replaceInvalidPredictions);
end
end

function [kTestMetrics, optIdx] = k_eval_usleep(trainIdx, testIdx, dataTable, method)

trainPred = dataTable{trainIdx,method};
trainPred = num2cell(horzcat(trainPred{:}),2);

trainTarget = dataTable{trainIdx,"target"};
trainTarget = horzcat(trainTarget{:});

kTrainMetrics = cellfun(@(x) calc_performance_metrics(trainTarget, x), trainPred);

% Find optimal threshold
[~, optIdx] = max(vertcat(kTrainMetrics.kappa));

testPred = dataTable{testIdx,method};
testPred = horzcat(testPred{:});
testPred = testPred(optIdx,:);

testTarget = dataTable{testIdx,"target"};
testTarget = horzcat(testTarget{:});

kTestMetrics = calc_performance_metrics(testTarget, testPred);

end