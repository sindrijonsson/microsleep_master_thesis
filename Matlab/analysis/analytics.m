TBL = table;
hz = 1;
step = 1;
for i = 1:numel(params.split.test)
id = string(params.split.train(i));

idx = master.id==id;
p=get_probs(fullfile(sprintf("predictions/%i_hz/",hz),sprintf("%s.mat",id)));
t = master.biWake_vs_biMSE(master.id==id);
tt = linspace(0, 40, 40*60*5);
tp = linspace(0, 40, 40*60*hz);

samps = (1:numel(t))';

if 1/step < hz
    
    resampidx = convert_target_time_to_prediction_samples(samps, step, hz);
    
    binProbs = arrayfun(@(x1,x2) {mean(p(x1:x2,:))}, resampidx(:,1), resampidx(:,2));
    binProbs = cell2mat(binProbs);

else
    binProbs = repelem(p,hz/step,1);
end

% figure(1); clf;
% ax1=subplot(3,1,1);
% plot_predictions(t,tt,"r",1,ax1);
% subplot(3,1,2);
% plot_probs(p, tp);
% subplot(3,1,3);
% plot_probs(binProbs, tt)
% binProbs = horzcat(binProbs, (1-binProbs(:,1)));
aasmPls = [AASM];
tbl = array2table(binProbs,"VariableNames",aasmPls);
tbl.target = t;
ds=stack(tbl,aasmPls,'NewDataVariableName','Probability');

TBL = [TBL; ds];
end
%%
TBL=TBL(~isnan(TBL.target),:);
TBL.target = ordinal(TBL.target);
b=boxchart(TBL.Probability_Indicator, TBL.Probability, ...
    "GroupByColor", TBL.target, "MarkerStyle","none");
legend(string([-1,1]))

%%
gTBL=grpstats(TBL,["Probability_Indicator","target"],["mean","sem"],"DataVars","Probability");

gTBL.x = repelem(1:5,1,2)';
figure(1); clf; hold on
for i = 1:height(gTBL)
    s=0.2;
    if mod(i,2); x = gTBL.x(i) - 0.2; else x = gTBL.x(i) + 0.2; end
    errorbar(x, gTBL.mean_Probability(i), gTBL.sem_Probability(i))
end
%%
TBL2 = TBL;
TBL2.State = ~(TBL.Probability_Indicator == "WAKE");
boxchart(TBL2.target, TBL2.Probability, "GroupByColor", TBL2.State)
legend(["Wake","Sleep"])

function samples = convert_target_time_to_prediction_samples(samples, step, prediction_hz)

t1 = [0; samples(1:end-1)*step];
t2 = samples*step;
maxPredictionSamples = (length(samples) * step) * prediction_hz;
samples=[max(1,floor(t1*prediction_hz)), min(ceil(t2*prediction_hz),maxPredictionSamples)];

end