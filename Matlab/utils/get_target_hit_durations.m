function [durCat, durHitCount, durTotalCount] = get_target_hit_durations(target, preds, secPerLabel)

if iscell(target); target=cell2mat(target); end
if iscell(preds); preds=cell2mat(preds); end

% Assign duration to the target labels
[f,l,s] = get_first_and_last(target, 1, false);
durs = ((l-f)+1)*secPerLabel;
targetAsDurs = nan(size(target));
for i = 1:numel(f)
    targetAsDurs(f(i):l(i)) = durs(i);
end
targetAsDurs(s) = secPerLabel;


% Find the durations where predictions hit the target
hitDurs = targetAsDurs(preds==1);

durCat = unique(targetAsDurs(~isnan(targetAsDurs)));
% Loop through the target durations and count total target and hit
durTotalCount = zeros(size(durCat));
durHitCount = zeros(size(durCat));
for i = 1:numel(durCat)
    durTotalCount(i) = sum(targetAsDurs==durCat(i));
    durHitCount(i) = sum(hitDurs==durCat(i));
end


end

%%
% Sort using categories
% catTarget = categorical(targetAsDurs(~isnan(targetAsDurs)));
% 
% catHit = categorical(hitDurs(~isnan(hitDurs)));
% 
% targetCategories = categories(catTarget)
% hitCategories = categories(catHit)
% 
% 
% targetCounts = countcats(catTarget)
% hitCounts = countcats(catHit)
% 
% hitPerc = hitCounts ./ targetCount