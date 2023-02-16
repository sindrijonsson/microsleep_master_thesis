function [Pxx, freq, tyy] = moving_pburg(y, order, fs, winSize, shiftSec, freqRes)


if nargin < 6; freqRes = nan; end
if nargin < 5; shiftSec = 0.200; end    % Overlap [s]
if nargin < 4; winSize = 1; end         % Window size [s]
if nargin < 3; fs = 200; end            % Sampling rate [Hz]
if nargin < 2; order = 16; end          % AR order


% Set parameters
winSamples = 1:(winSize * fs);
shiftSamples = shiftSec * fs;
len = length(y);
if ~isnan(freqRes)
    nfft = fs/freqRes;
    numSamp = nfft/2 + 1;
    fcn = @(x) pburg(x,order,nfft);
else
    nfft = 256/2 + 1;
    numSamp = nfft;
    fcn = @(x) pburg(x, order);
end

numWins = (len / shiftSamples) - ceil((fs / shiftSamples) - 1);
Pxx = zeros(numSamp, numWins);
midWindow = winSize / 2;
tyy = midWindow:shiftSamples/fs:(len/fs)-midWindow;


i = 1;
while max(winSamples) <= length(y)
    [pxx, f] = fcn(y(winSamples));
    Pxx(:,i) = pxx;
    i = i + 1;
    winSamples = winSamples + shiftSamples;
end

if max(f) <= pi
    freq = f/pi*100;
else
    freq = f;
end

end