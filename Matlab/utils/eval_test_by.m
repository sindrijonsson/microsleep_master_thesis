function [overallMetrics, perRecMetrics] = eval_test_by(testResults, by, params)

assert(ismember(lower(by), ["event","sample"]), ...
       "By argument must be either event or sample!!!");

if strcmp(by, "event")
    evalFcn = @(yTrue, yHat) calc_hit_stats(yTrue, yHat);
else 
    evalFcn = @(yTrue, yHat) calc_performance_metrics(yTrue, yHat, params);
end


perRecMetrics = cellfun(@(x,y) evalFcn(x,y), ...
                        testResults.yTrue, testResults.yHat);

perRecMetrics = struct2table(perRecMetrics);
perRecMetrics.id = testResults.id;

% Predict on concatenated data

try 
overallMetrics = evalFcn(...
                        cell2mat(testResults.yTrue), ...
                        cell2mat(testResults.yHat) ...
                        );
catch ME
    if strcmp(ME.identifier, 'MATLAB:catenate:dimensionMismatch')
        overallMetrics = evalFcn(...
                                 cell2mat(testResults.yTrue'), ...
                                 cell2mat(testResults.yHat') ...
                                 );
    else
        throw(ME)
    end
    
end

end