function stats = calc_hit_stats(yTrue, yHat, hdl)

if nargin < 3; hdl=0; end

[tp, fp, fn] = calc_hits_with_singles(yTrue, yHat);

% Precision
if (tp + fp) > 0
    precision = tp / (tp + fp);
else
    precision = 0;
end

% Recall
if (tp + fn) > 0
   recall = tp / (tp + fn);
else
   recall = 0;
end

% F1 score
if (precision + recall) > 0
   f1 = (2 * precision * recall) / (precision + recall);
else
   f1 = 0;
end

stats=struct;
stats.precision = precision;
stats.recall = recall;
stats.f1 = f1;

   
end