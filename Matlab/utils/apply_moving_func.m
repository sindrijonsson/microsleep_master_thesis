function out = apply_moving_func(y, fs, winSizeSec, winStepSec, fcn)

% Set parameters
winSamples = 1:(winSizeSec * fs);
shiftSamples = winStepSec * fs;
len = length(y);
numWins = (len / shiftSamples) - ceil((fs / shiftSamples) - 1);
out = nan(size(y,1), numWins);

i = 1;
while max(winSamples) <= len
    
    out(:,i) = fcn(y(:,winSamples));
    winSamples = winSamples + shiftSamples;
    i = i+1;
end

end