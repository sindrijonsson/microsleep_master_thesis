function  [ax] = make_skorucack_label_plots(labels, time, ax)

if nargin < 3
    figure(1);
    clf
    ax=gca;
end

fn = flip(fieldnames(labels));
for i = 1:height(fn)
    field = string(fn(i));
    label = labels.(field);
    [first, last, singles] = get_first_and_last(label, 1);
    plot_patches(first, last, i-1, i, time, [1,0,0], 1, ax)
    if any(singles)
        plot(ax, [time(singles); time(singles)], ...
             [repmat(i-1,size(singles)); repmat(i, size(singles))], "Color", [1, 0, 0], "LineWidth", 1)

    end
end
ax.set("YTick",0.5:1:4.5);
ax.set("YTickLabels",fn);
ylim([0,5])
box on

% monitorPositions = get(groot,"MonitorPositions");
% if size(monitorPositions,1) == 3
%     fig=gcf;
%     fig.set("Position",[2125,365, 1598, 328])
% end
xlabel("time (min)")
end