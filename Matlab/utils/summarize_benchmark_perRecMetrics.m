function [m,s] = summarize_benchmark_perRecMetrics(tbl, by)

assert(ismember(lower(by), ["event","sample"]), ...
       "By argument must be either event or sample!!!");

sem = @(x) std(x,"omitnan") / sqrt(sum(~isnan(x)));

if strcmpi(by, "sample")
    
    % Calculate mean scores
    m = struct;
    
    m.specificity = mean(tbl.specificity);
    m.accuracy    = mean(tbl.accuracy);
    m.fpMinutes   = mean(tbl.fpMinutes);
 
    % Handle nans
    m.recall      = mean(tbl.recall,"omitnan");
    m.precision   = mean(tbl.precision,"omitnan");
    m.kappa          = mean(tbl.kappa,"omitnan");
    
    % Calculate standard error of means
    s = struct;
    
    s.specififty  = sem(tbl.specificity);
    s.accuracy    = sem(tbl.accuracy);
    s.fpMinutes   = sem(tbl.fpMinutes);
    s.recall      = sem(tbl.recall);
    s.precision   = sem(tbl.precision);
    s.kappa          = sem(tbl.kappa);

else

    % Remove entries where every value is zero
    arrTable = table2array(tbl(:,["precision","recall","kappa"]));
    validIdx = all(arrTable, 2);
    tbl = tbl(validIdx,:);

    % Calculate mean scores
    m = struct;
    
    m.precision = mean(tbl.precision,"omitnan");
    m.recall    = mean(tbl.recall,"omitnan");
    m.kappa        = mean(tbl.kappa,"omitnan");
    
    % Calculate standard error of means
    s = struct;

    s.precision = sem(tbl.precision);
    s.recall    = sem(tbl.recall);
    s.kappa        = sem(tbl.kappa);

end

        

end