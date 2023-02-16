function view_rec(rec, pltSpectra)

numSubs = 6;
if nargin < 2 || pltSpectra == 0
    pltSpectra = 0;
    numSubs = 4;
end


Data = load(rec);
Data = Data.Data;

eeg = table;
eeg.O1 = Data.eeg_O1;
eeg.O2 = Data.eeg_O2;
eeg.E1 = Data.E1;
eeg.E2 = Data.E2;

fs = Data.fs;

time = linspace(0, height(eeg)/(fs), height(eeg));
xl = [0, 2400];
eeg.time = time';
%

vars = ["E1", "E2", "O1", "O2"];

figure(1); clf;
set(gcf,"WindowState","maximized")

axs = gobjects(numSubs,1);

for i = 1:numel(vars)

    var = vars(i);
    subplot(numSubs,1,i);
    ax = gca;
    hold on;
    p=plot(eeg.time, eeg{:,var});
    axis tight
    box on
    xlim(xl)
    xlabel("Time [min]");
    ylabel(strcat(var, " (\muV)"));



    if i < numSubs
        xlabel("")
        xticklabels([])
    end

    if var == "O1"
        mse = find(Data.labels_O1 == 1)';
        msec = find(Data.labels_O1 == 2)';
        ed = find(Data.labels_O1 == 3)';

    elseif var == "O2"
        mse = find(Data.labels_O2 == 1)';
        msec = find(Data.labels_O2 == 2)';
        ed = find(Data.labels_O2 == 3)';
    end

    if ismember(var,["O1","O2"])
        mseLegend = set_labels(ax, 'g', time, mse);
        msecLegend = set_labels(ax, 'b', time, msec);
        edLegend = set_labels(ax, 'r', time, ed);

        labAxPos = get(ax,"Position");
        legend( ax, ...
            [mseLegend, msecLegend, edLegend], ...
            ["MSE","MSEc","ED"], "Location","EastOutside" )
        set(ax,"Position",labAxPos);
    end

    axs(i) = ax;

    drawnow;
end


% Add spectra if wanted

sVars = vars(3:end);

if pltSpectra

    for j = 1:numel(sVars)
        var = sVars(j);
        [pyy, freq, tyy] = get_spectra(eeg{:,var});

        subplot(numSubs, 1, i+j)
        imagesc(tyy, freq, pyy);
        ax = gca;
        axs(i+j) = ax;
        axis xy
        colormap jet
        xlabel("Time [min]")
        ylabel(sprintf("%s (Hz)",var));

        cAxPos = get(ax,"Position");
        cb=colorbar(ax, "eastoutside");
%         cb.Limits([-20m ])
        caxis([-20 25]);
        set(ax,"Position",cAxPos);
        xlim(xl)
        if i+j < numSubs
            xlabel("")
            xticklabels([])
        end
    end
    set(gcf, 'Name', 'EEG Viewer')

end

linkaxes(axs(1:2), 'y');
% linkaxes(axs(3:4), 'y');
% if pltSpectra; linkaxes(axs(5:6),'y'); end
linkaxes(axs, 'x');
arrayfun(@(x) set(x,"YTickLabelMode","auto"), axs);
arrayfun(@(x) set(x,"YLimMode","auto"), axs);

% addlistener(axs, {'XLim'}, 'PostSet', ...
%             @(src, eventData) update_y(src, eventData, axs(1:4)));

% Beautify
set(findall(gcf,'-property','FontSize'),'FontSize',12)
sgtitle(rec,"FontSize",16,"FontWeight","bold")

end
%%
function out = set_labels(ax, col,  time, idx)

if isempty(find(idx, 1))
    out = fill([nan,nan,nan,nan],[nan,nan,nan,nan],col,...
        'FaceAlpha',0.15, 'EdgeColor','none');
    return
end

D = diff([0,diff(idx)==1, 0]);
first = idx(D>0);
last = idx(D<0);
yMin = ax.YLim(1); yMax = ax.YLim(2);
for i = 1:numel(first)
    f = first(i); l = last(i);
    out = fill(ax, ...
        [time(f) time(l) time(l) time(f)], ...
        [yMin, yMin, yMax, yMax], ...
        col, ...
        'FaceAlpha',0.15, ...
        'EdgeColor','none');
end

end

% function [] = update_y(~, eventData, axs)
% time = eventData.AffectedObject.Children.XData;
% newXLim = eventData.AffectedObject.XLim;
% newXIdx = [newXLim(1):newXLim(2)];
% 
% ii = dsearchn(time',newXIdx');
% disp("Zoom")
% for i = 1:numel(axs)
%     i
%     ax = axs(i);
%     ch = findall(ax,'-property','Children','type','line');
%     ylim(ax, [min(ch.YData(ii))*1.1 max(ch.YData(ii))*1.1]);
% end
% end