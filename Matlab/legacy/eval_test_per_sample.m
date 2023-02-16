function [overallMetrics, perRecMetrics] = eval_test_per_sample(testResults, params)



perRecMetrics = cellfun(@(x,y) calc_performance_metrics(x,y,params), ...
                        testResults.yTrue, testResults.yHat);

perRecMetrics = struct2table(perRecMetrics);
perRecMetrics.id = testResults.id;

% Predict on concatenated data


try 
overallMetrics = calc_performance_metrics(...
                                          cell2mat(testResults.yTrue), ...
                                          cell2mat(testResults.yHat), ...
                                          params ...
                                          );
catch ME
    if strcmp(ME.identifier, 'MATLAB:catenate:dimensionMismatch')
        overallMetrics = calc_performance_metrics(...
                                          cell2mat(testResults.yTrue'), ...
                                          cell2mat(testResults.yHat'), ...
                                          params ...
                                          );
    else
        throw(ME)
    end
    
end

end