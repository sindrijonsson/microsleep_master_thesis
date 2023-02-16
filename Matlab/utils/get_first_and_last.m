function [first, last, singles] = get_first_and_last(y, target, skipSingles)


if nargin < 3
    skipSingles = 0;
end

if size(y,1) > size(y,2)
    y = y';
end

idx = find(y==target);
D = diff([0,diff(idx)==1, 0]);
first = idx(D>0);
last = idx(D<0);

%     % Find singles
%     intervals = arrayfun(@(x,y) x:y, first,last,'UniformOutput',false);
%     intervals = horzcat(intervals{:});
%
%     % Singles
%     singles = idx(~ismember(idx, intervals));

if skipSingles
%     fprintf("Skipping Singles!!!\n")
    singles = [];
else
    y = y == target;
    singles = strfind([0 y 0], [0 target 0]);
end


end

