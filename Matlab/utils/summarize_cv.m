function cvStats = summarize_cv(cv, params)


eval_metric = params.usleep.evalMetric;

cvStats = grpstats(cv,["method","hz"],["mean","sem"],...
    "DataVars",eval_metric);
cvStats.Properties.RowNames = {};
hz = unique(cvStats.hz);
m = unique(cvStats.method);
binM = contains(m,"Bin");
m = [m(~binM); m(binM)];
offset = linspace(-1,1,length(m));
figure(1);clf;hold on
cc=lines(length(m));
x=linspace(2,20,4);
for i = 1:numel(hz)
    lhdl=[];
    for j = 1:numel(m) 
        tmp = cvStats((cvStats.hz == hz(i) & cvStats.method == m(j)),:);
        if ~isempty(tmp) 
            p=plot(x(i)+offset(j), tmp.("mean_"+eval_metric), ...
                     "Marker","o","MarkerFaceColor",cc(j,:),"MarkerEdgeColor",cc(j,:));
            errorbar(x(i)+offset(j), tmp.("mean_"+eval_metric), tmp.("sem_"+eval_metric), ...
            "Marker","none","Color",cc(j,:));
        end
    lhdl = [lhdl, p];
    end
end
xlim([0, 22]);
xticks(x);
xticklabels(hz);
xlabel("Hz")
ylabel(sprintf("%s (+/- SEM)",upper(eval_metric)));
set(findall(gca,"Type","Line"),"MarkerSize",8)
set(findall(gca,"Type","ErrorBar"),"LineWidth",2)
legend(lhdl, m, "NumColumns",3,"Location","southeast")
grid on
box on
set(findall(gcf,'-property','FontSize'),'FontSize',14)
[~,optIdx] = max(cvStats.mean_f1);
optHz = cvStats{optIdx,"hz"};
optMethod = cvStats{optIdx,"method"};
optCV=cvStats(cvStats.hz==optHz & cvStats.method==optMethod,:);
title(sprintf("Optimal model: %s @ %i Hz = %.2f", ...
                              optMethod, optHz, optCV.("mean_"+eval_metric)))

tblFile = fullfile(params.outFolder, "cv_stats.csv");
writetable(cvStats,tblFile);

end