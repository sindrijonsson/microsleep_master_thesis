function [TP, FP, FN] = calc_hits_with_singles(yTrue, yHat)


[tStart, tStop, tSingle] = get_first_and_last(yTrue,1);
[pStart, pStop, pSingle] = get_first_and_last(yHat, 1);

% Add singles
tStart = [tStart, tSingle]; tStop = [tStop, tSingle];
pStart = [pStart, pSingle]; pStop = [pStop, pSingle]; 

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
