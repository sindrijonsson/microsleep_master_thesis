function [overallMetrics, perRecMetrics] = eval_test_per_event(testResults)


perRecMetrics = cellfun(@(x,y) calc_hit_stats(x,y), ...
                        testResults.yTrue, testResults.yHat);

perRecMetrics = struct2table(perRecMetrics);
perRecMetrics.id = testResults.id;

% Predict on concatenated data

try 
overallMetrics = calc_hit_stats(...
                                  cell2mat(testResults.yTrue), ...
                                  cell2mat(testResults.yHat) ...
                                  );
catch ME
    if strcmp(ME.identifier, 'MATLAB:catenate:dimensionMismatch')
        overallMetrics = calc_hit_stats(...
                                          cell2mat(testResults.yTrue'), ...
                                          cell2mat(testResults.yHat') ...
                                          );
    else
        throw(ME)
    end
    
end

end