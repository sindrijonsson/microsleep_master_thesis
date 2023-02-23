tests = {testMuSleep, testUsleep, testSSL, testLSTM, testRF, testSVM, testMalafeev};

fpTable = table;
for i = 1:numel(tests)
    tmp = tests{i};
    durFP = cellfun(@(t, p) get_duration_false_positives(t, p, params), ...
                    tmp.yTrue, tmp.yHat, ...
                    "UniformOutput",false);
    durFP = cell2mat(durFP);
    t = table;
    t.dur = durFP(:,1);
    t.fp  = durFP(:,2);
    
    stats = rowfun(@(fp) sum(fp)/length(fp), t, ...
                    "InputVariables",["fp"], ...
                    "GroupingVariables","dur", ...
                    "OutputVariableNames","perc");
    stats.model = repmat(mdls(i),height(stats),1);

    fpTable = [fpTable; stats];
end

tStats=unstack(fpTable, ["perc"], "model","VariableNamingRule","preserve");
tStats=tStats(:,["dur","GroupCount",mdls']);

figure(2); clf; hold on;

figure(3); clf; hold on;


hdl = [];
hh = [];
bhdl = [];
uniX = unique(fpTable.dur);
cc=lines(length(unique(fpTable.model)));
for i = 3:size(tStats,2)
    tmp = tStats(:,[1,2,i]);
    tmp = tmp(~isnan(tmp{:,end}),:);
    x=tmp.dur;
    y=table2array(tmp(:,end));
    [x, sx] = sort(x);
    y = y(sx);
    idx = [1, 3; 3, 15; 16, find(x==max(x))];
    
    c=cc(i-2,:);

    ylim([0, 1.05])
    figure(2); hold on;
    ix = ismember(uniX,x);
    px = categorical(uniX);
    py = nan(size(px));
    py(ix) = y;
    ax1=subplot(numel(mdls),1,i-2); hold on;
    plot(x,y, "o-","Color",c,"LineWidth",2,"MarkerSize",2);
    h=bar(x,y,"EdgeColor","none","FaceColor",c,"FaceAlpha",0.2);
%     h=area(px,py,"FaceColor",c,"EdgeColor","none");
    hh = [hh; h];
    box on
    grid on
    ylabel(tmp.Properties.VariableNames(end), ...
        "Interpreter","none","Rotation",45, ...
        "HorizontalAlignment","right" )
    if i < size(tStats,2)
%         if h.t
        if h.Type == "bar"; xticks(0:50:300); end
        xticklabels([]);
    end

    figure(3); hold on;
    ax2 = gca;
    my = arrayfun(@(x1,x2) mean(y(x1:x2)), idx(:,1), idx(:,2));
    b=bar(ax2,(1:3)+(0.1*i), my, 0.1);
    grid("minor")
    bhdl = [bhdl,b];
    xticks(1.5:1:3.5);
    xticklabels(["1 - 3","3 - 15","15 +"]);
    yt = 0:0.2:1;
    yticks(yt);
    yticklabels(string(yt));
    grid on
    ylabel("Mean % FP")
    xlabel("Duration [s]")
    box on
end



legend(bhdl, tStats.Properties.VariableNames(3:end), ...
    "Location","northeast","Interpreter","none")

for j = 1:numel(hh)
ax = hh(j).Parent;
xline(ax,3,"--","LineWidth",2)
xline(ax,15,"--","LineWidth",2)
end

figure(2);
sgtitle("%FP as a function of prediction duration")

axEnd = hh(end).Parent;
xlabel(axEnd,"Duration [s]")

figure(2);
linkaxes(findall(gcf,"Type","Axes"),'xy')

set(findall(gcf,"-property","FontSize"),"FontSize",16)

% fig = figure(2);
% xlim([0 200])
% set(fig,'position',[fig.Position(1:2), 1280, 1280])
% figFile = fullfile(params.outFolder,"figures","fp_fig_long.png");
% exportgraphics(fig, figFile, "Resolution",300);

function out = get_duration_false_positives(targets, preds, params)

% Get the durations of the predictions
predDurs = get_durations(preds, params.secPerLabel);

% Get the indexes of the predictions
[predStart, predStop, predSingle] = get_first_and_last(preds, 1, false);

% Create array of durations of predictions (non predictions are nan)
d = nan(size(preds));

for i = 1:numel(predStart)
    d(predStart(i):predStop(i)) = predDurs(i);
end

d(predSingle) = params.secPerLabel;

% Set indexes where they are correct
fp = (preds == 1) & (targets == -1);

if size(d,2) > size(d,1)

    out = [d', fp'];    
else
    out = [d, fp];
end
out = out(~isnan(out(:,1)),:);

end