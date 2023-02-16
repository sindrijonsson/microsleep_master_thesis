function plot_patches(first, last, yMin, yMax, time, col, alpha, ax)

if nargin < 7; alpha = 1; end
if nargin < 8; ax = gca; end

hold on

if all(col==0); edgeCol = "none"; else; edgeCol=col; end


for i = 1:numel(first)
    f = first(i); l = last(i);
    fill(ax, ...
        [time(f) time(l) time(l) time(f)], ...
        [yMin, yMin, yMax, yMax], ...
        col, ...
        'FaceAlpha',alpha, ...
        'EdgeColor',edgeCol);
end
xlim([min(time),max(time)])
end