function [TP, FP, FN] = calc_hits(yTrue, yHat)


% BEWARE WE ARE NOT USING SINGLES IN CASE THAT METRIC BECOMES NECESSARY
% E.g. when calculating performance when targets and predictions have no
% duration constraints

[tStart, tStop, ~] = get_first_and_last(yTrue,1);
[pStart, pStop, ~] = get_first_and_last(yHat, 1);

tScored = zeros(1,length(tStart));
pScored = zeros(1,length(pStart));

for i = 1:numel(pStart)
    
    match = ((tStop >= pStart(i)) & (pStop(i) >= tStart));
    
    if any(match)
        tScored(match) = 1;
        pScored(i) = 1;
    else
        pScored(i) = 0;
    end
end

TP = sum(pScored==1);
FP = sum(pScored==0);
FN = sum(tScored==0);

end
