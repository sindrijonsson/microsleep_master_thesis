function out = remove_invalid_labels(y, minDur, maxDur, secPerSamp, invalidValue, skipSingles)

if nargin < 2; minDur = 0; end
if nargin < 3; maxDur = inf; end
if nargin < 4; secPerSamp = 0.2; end          % Default to 5 Hz
if nargin < 5; invalidValue = nan; end
if nargin < 6; skipSingles = false; end


[first, last, singles] = get_first_and_last(y, 1, skipSingles);
samps = (last - first) + 1;
dur = samps * secPerSamp;
invalidDurations = (dur < minDur) | (dur > maxDur);
invalidFirst = first(invalidDurations);
invalidLast = last(invalidDurations);

out = y;
for i = 1:numel(invalidFirst)
    out(invalidFirst(i):invalidLast(i)) = invalidValue;
end

out(singles) = invalidValue;

end