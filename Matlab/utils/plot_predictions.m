function  [ax] = plot_predictions(y, time, col, alpha, ax)


if nargin < 3; col = "r"; end
if nargin < 4; alpha = 1; end
if nargin < 5
    figure(1);
    clf
    ax=gca;
end

[first, last, singles] = get_first_and_last(y, 1);
plot_patches(first, last, 0, 1, time, col, alpha, ax)
if any(singles)
    try
        axline(ax, time(singles), col)
    catch
        plot(ax, [time(singles); time(singles)], ...
            [zeros(size(singles)); ones(size(singles))], "Color", col, "LineWidth", 1)
    end
end
box on
yticks([])
yticklabels([])
end