function out = convert_labels_to_classes(y)
    
    out = struct;
    out.W    = parse_bern(y, 0, @all);
    out.ED   = parse_bern(y, 3, @any);
    out.MSEc = parse_bern(y, 2, @any);

    % slightly different method for finding unilateral MSE (MSEu)
    arr = zeros(1,size(y,2)); % changing this to zeros changes the number of ms
    MSEu = y(1,:) == 1 & y(2,:) ~= 1 | y(1,:) ~= 1 & y(2,:) == 1;
    arr(MSEu) = 1;
    out.MSEu = arr;

    out.MSE = parse_bern(y, 1, @all);

end

function out = parse_bern(y, target, criteria, val)
    
    if nargin < 4; val = 1; end

    out = zeros(1, size(y,2));
    idx = criteria(y == target);
    out(idx) = val;

end