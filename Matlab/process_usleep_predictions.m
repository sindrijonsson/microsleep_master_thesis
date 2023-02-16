function uSleep = process_usleep_predictions(params, master)


% -------------------------------------------------------------------------

secPerLabel = params.secPerLabel;
labelsPerSec = params.labelsPerSec;
trainTargets = params.trainTargets;
testTargets = params.testTargets;
negClassValue = params.negClassValue;
split = params.split;

HZ = params.usleep.hz;
thresholds = params.usleep.thresholds;
splitValue = params.usleep.splitValue;
outFile = params.usleep.outfile;

% -------------------------------------------------------------------------

uSleep = table;
names = unique(master.id);

for hIdx = 1:numel(HZ)
    hz = HZ(hIdx);
    for nIdx = 1:numel(names)

        name = names(nIdx);

        p = sprintf("predictions\\%i_hz\\%s",hz,name);
        fprintf("Processing U-Sleep %s\n",p);
        tmp=load(p);
        tmp = double(tmp.data);

        % Get reference data from master
        masterRef=master(master.id == name,:);

        % Get indexs for resampling the predictions to match the targets
        resampIdxs = convert_target_time_to_prediction_samples(masterRef.sample, secPerLabel, hz);

        % convert cumulative probabilties to 0 - 1
        probs = tmp ./ sum(tmp,2);

        % resample probabilites
        probsResampled = iter_bin_calc(probs', resampIdxs, @(x) mean(x,2));

        if hz > labelsPerSec

            % preds N1
%             probsN1 = probs(:, 2)';
%             predsN1 = double(probsN1 >= thresholds);
%             predsN1 = iter_resample(predsN1, resampIdxs, params);
%             predsN1 = iter_remove_invalid_labels(predsN1, params);
%             predsN1(predsN1 == 0) = negClassValue;
% 

            % preds argmax
            [~, aasmPredsArgmax] = max(probs,[],2);
            predsArgmax = aasm_to_wake_vs_sleep(aasmPredsArgmax)';
            predsArgmax = iter_resample(predsArgmax, resampIdxs, params);
            predsArgmax = iter_remove_invalid_labels(predsArgmax, params);
            predsArgmax(predsArgmax == 0) = negClassValue;

            % probs sum
            probsSum = sum(probs(:,2:end),2)';
            predsSum = double(probsSum >= thresholds);
            predsSum = iter_resample(predsSum, resampIdxs, params);
            predsSum = iter_remove_invalid_labels(predsSum, params);
            predsSum(predsSum==0) = negClassValue;
    
            % probs max
            probsMax = max(probs(:,2:end),[],2)';
            predsMax = double(probsMax >= thresholds);
            predsMax = iter_resample(predsMax, resampIdxs, params);
            predsMax = iter_remove_invalid_labels(predsMax, params);
            predsMax(predsMax==0) = negClassValue;

            % Using mean probabilities within bin for argmax
            probsBin = probsResampled;
            [~, predsBinArgmax] = max(probsBin, [], 1);
            predsBinArgmax = aasm_to_wake_vs_sleep(predsBinArgmax);
            predsBinArgmax = iter_remove_invalid_labels(predsBinArgmax, params);
            predsBinArgmax(predsBinArgmax == 0) = negClassValue;
    
            % Using the mean within bin for sum of sleep probabilites
            probsBinSum = iter_bin_calc(probsSum, resampIdxs, @(x) mean(x,2));
            predsBinSum = double(probsBinSum >= thresholds);
            predsBinSum = iter_remove_invalid_labels(predsBinSum, params);
            predsBinSum(predsBinSum==0) = negClassValue;
    
            % Using the mean within bin for the max sleep probability
            probsBinMax = iter_bin_calc(probsMax, resampIdxs, @(x) mean(x,2));
            predsBinMax = double(probsBinMax >= thresholds);
            predsBinMax = iter_remove_invalid_labels(predsBinMax, params);
            predsBinMax(predsBinMax==0) = negClassValue;
        
            
        else
        
            % preds N1
%             probsN1 = probs(:, 2)';
%             predsN1 = double(probsN1 >= thresholds);
%             predsN1 = repelem(predsN1, 1, labelsPerSec);
%             predsN1 = iter_remove_invalid_labels(predsN1, params);
%             predsN1(predsN1 == 0) = negClassValue;


            % preds argmax
            [~, aasmPredsArgmax] = max(probs,[],2);
            predsArgmax = aasm_to_wake_vs_sleep(aasmPredsArgmax)';
            predsArgmax = repelem(double(predsArgmax), 1, labelsPerSec);
            predsArgmax = iter_remove_invalid_labels(predsArgmax, params);
            predsArgmax(predsArgmax == 0) = negClassValue;

            % probs sum
            probsSum = sum(probs(:,2:end),2)';
            predsSum = double(probsSum >= thresholds);
            predsSum = repelem(predsSum, 1, labelsPerSec);
            predsSum = iter_remove_invalid_labels(predsSum, params);
            predsSum(predsSum==0) = negClassValue;
    
            % probs max
            probsMax = max(probs(:,2:end),[],2)';
            predsMax = double(probsMax >= thresholds);
            predsMax = repelem(predsMax, 1, labelsPerSec);
            predsMax = iter_remove_invalid_labels(predsMax, params);
            predsMax(predsMax==0) = negClassValue;
    
            % ignore bin methods
            predsBinArgmax = [];
            predsBinSum = [];
            predsBinMax = [];
            
        end

        
        % Collect into table
        entry = struct;
        compress = @(x) mat2cell(x,size(x,1),size(x,2));
        entry.id                = name;
        entry.train             = ismember(name,split.train);
        entry.hz                = hz;
        entry.trainTargets      = compress(masterRef{:,trainTargets}');
        entry.testTargets       = compress(masterRef{:,testTargets}');
        entry.probs             = compress(probsResampled);
        entry.predsArgmax       = compress(predsArgmax);
        entry.predsSum          = compress(predsSum);
        entry.predsMax          = compress(predsMax);
        entry.predsBinArgmax    = compress(predsBinArgmax);
        entry.predsBinSum       = compress(predsBinSum);
        entry.predsBinMax       = compress(predsBinMax);
%         entry.predsN1       = compress(predsN1);

        % Append
        uSleep = [uSleep; struct2table(entry)];
    end
end
save(outFile, "uSleep");

end

function out = aasm_to_wake_vs_sleep(probs)
out=probs;
out(probs==1) = 0;
out(probs~=1) = 1;
end

function samples = convert_target_time_to_prediction_samples(samples, step, prediction_hz)

t1 = [0; samples(1:end-1)*step];
t2 = samples*step;
maxPredictionSamples = (length(samples) * step) * prediction_hz;
samples=[max(1,floor(t1*prediction_hz)), min(ceil(t2*prediction_hz),maxPredictionSamples)];

end

function out = iter_resample(y, reIdx, params)

out = nan(size(y,1),size(reIdx,2));

for i = 1:height(reIdx)
    out(:,i) = params.downsampleFcn(y(:,reIdx(i,1):reIdx(i,2)));
end

out(out==params.usleep.splitValue) = 1;

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
                                    params.windowSizeSec, params.replaceInvalidPredictions, ...
                                    params.skipSingles);
end

end
