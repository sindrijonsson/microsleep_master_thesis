function [pyy, freq, tyy] = get_spectra(y, order, fs, freq, winSize, shiftSec)

if nargin < 6; shiftSec = 0.200; end            % Overlap [s]
if nargin < 5; winSize = 1; end                 % Window size [s]
if nargin < 4; freq = 0:0.2:30; end            % Frequency range [Hz]
if nargin < 3; fs = 200; end                    % Sample rate [Hz]
if nargin < 2; order = 16; end                  % AR order 

% Set parameters
winSamples = 1:(winSize * fs);
shiftSamples = shiftSec * fs;
len = length(y);
numWins = (len / shiftSamples) - ceil((fs / shiftSamples) - 1);
pyy = zeros(length(freq), numWins);
midWindow = winSize / 2;
tyy = midWindow:shiftSamples/fs:(len/fs)-midWindow;

i = 1;
y=y';
while max(winSamples) <= length(y)
    H = pburg(y(winSamples), order, freq, fs);
    pyy(:,i) = pow2db(H);
    i = i + 1;
    winSamples = winSamples + shiftSamples;
end


end