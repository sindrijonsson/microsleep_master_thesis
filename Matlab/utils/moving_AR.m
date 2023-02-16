function [pyy, freq, tyy] = moving_AR(y, order, freqRange, winSize, shiftSec)

if nargin < 5; shiftSec = 0.200; end            % Overlap [s]
if nargin < 4; winSize = 1; end                 % Window size [s]
if nargin < 3; freqRange = 0:0.02:26; end       % Frequency range [Hz]
if nargin < 3; fs = 200; end                    % Sample rate [Hz]
if nargin < 2; order = 16; end                  % AR order 

% Set parameters
winSamples = 1:(winSize * fs);
shiftSamples = shiftSec * fs;
len = length(y);
numWins = (len / shiftSamples) - ceil((fs / shiftSamples) - 1);
pyy = zeros(length(freqRange), numWins);
midWindow = winSize / 2;
tyy = midWindow:shiftSamples/fs:(len/fs)-midWindow;

i = 1;
while max(winSamples) <= length(y)
    [p, f] = pburg(y(winSamples), order, freqRange, fs);
    pyy(:,i) = p;
    i = i + 1;
    winSamples = winSamples + shiftSamples;
end
freq = f;

end