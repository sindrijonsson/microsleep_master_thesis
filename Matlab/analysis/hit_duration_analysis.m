
r=get_target_hit_duration_table(testRF,params);
s=get_target_hit_duration_table(testSVM,params);
l=get_target_hit_duration_table(testLSTM,params);
u=get_target_hit_duration_table(testUsleep,params);
m=get_target_hit_duration_table(testMalafeev,params);
mu=get_target_hit_duration_table(testMuSleep,params);
sl=get_target_hit_duration_table(testSSL,params);

stats = [mu;u;sl;l;r;s;m];
m = mdls;
stats.model = repelem(m,height(r),1);
stats.dur = stats.dur;

% figure(1); clf; hold on;
% rowfun(@(x,y) fill(x, y, "-"), stats, ...
%     "InputVariables",["dur","perc"], ...
%     "GroupingVariables","model");

tStats=unstack(stats, "perc", "model",'VariableNamingRule','preserve');
tStats=tStats(:,["dur","GroupCount",m']);

figure(2);clf; hold on;
hdl = [];
hh = [];
bhdl = [];

x=tStats.dur;
idx = [1, 3; 3, 15; 16, find(x==max(x))];
cc=lines(length(unique(stats.model)));
for i = 3:size(tStats,2)
    y=table2array(tStats(:,i));
    c=cc(i-2,:);
    subplot(3,1,[1,2]); hold on;
    h=plot(ordinal(x),y, ".", ...
        "MarkerSize",8, ...
        "Color",c);
    hdl=[hdl, h];
    
    xticks(1:length(tStats.dur))
    xticklabels(tStats.dur)
    box on
    xlim([0, length(categories(ordinal(x)))+1])
    ylim([0, 1.05])
    
    subplot(3,1,[1,2]); hold on;
    h=plot(movmean(y,5), "-","Color",c,"LineWidth",3);
    hh = [hh; h];
    xticks(1:length(tStats.dur))
    xticklabels(tStats.dur)
    xlabel("Duration [s]")
    box on
%     axis tight
    grid on
    ylabel("Recall")

    subplot(3,1,3); hold on;
    my = arrayfun(@(x1,x2) mean(y(x1:x2)), idx(:,1), idx(:,2));
    b=bar((1:3)+(0.1*i), my, 0.1);
    grid("minor")
    bhdl = [bhdl,b];
    xticks(1.5:1:3.5);
    xticklabels(["1 - 3","3 - 15","15 +"]);
    yt = 0:0.2:1;
    yticks(yt);
    yticklabels(string(yt));
    grid on
    ylabel("Recall")
    xlabel("Duration [s]")
    box on
end

% ix=unique(idx)
% for i = 1:2
%     subplot(3,1,i)
%     for j = 1:numel(ix)
%     xline(ix(j),'--')
%     end
% end

legend(hdl, tStats.Properties.VariableNames(3:end), ...
    "Location","southeast","AutoUpdate","off","Interpreter","none")
legend(hh, tStats.Properties.VariableNames(3:end), ...
    "Location","southeast")
legend(bhdl, tStats.Properties.VariableNames(3:end), ...
    "Location","northeast","Interpreter","none")

ax = hdl(1).Parent;
xline(ax,3,"--","LineWidth",2)
xline(ax,15,"--","LineWidth",2)
sgtitle("Recall as a function of MS duration")

set(findall(gcf,"-property","FontSize"),"FontSize",16)

%%
% rowfun(@(x) mean(x), stats, "InputVariables","perc", "GroupingVariables","model")   