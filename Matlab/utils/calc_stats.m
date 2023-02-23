function stats = calc_stats(x, nanHdl)

if nargin < 2; nanHdl = "includenan"; end

stats = struct;
stats.count  = numel(x);
stats.mean   = mean(x,nanHdl);
stats.min    = min(x, [], nanHdl);
stats.prc25  = prctile(x, 25);
stats.median = median(x, nanHdl);
stats.prc75  = prctile(x, 75);
stats.max    = max(x, [], nanHdl);
stats.within = sum((x >= 3) & (x <= 15)) / length(x);

% Calc # samples per duration
[cnt, val] = hist(x, unique(x));
total = sum(x);
stats.sampLess3 = sum(cnt(val<3) .* val(val<3)) / total;
stats.sampWithin = sum(cnt(val>=3 & val<=15) .* val(val>=3 & val<=15)) / total;
stats.sampGreater15 = sum(cnt(val>15) .* val(val>15)) / total;


end