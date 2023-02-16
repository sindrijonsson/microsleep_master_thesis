target = testUsleep{testUsleep.id=="tG6i","yTrue"}{1};
[tStart, tStop, ~] = get_first_and_last(target,1);

preds = zeros(size(target));

% Create test predictions 1
preds1 = preds;
preds1(target==1) = 1;

% Remove first prediction
preds1(tStart(1):tStop(1)) = 0;

figure(1); clf;
ax=subplot(2,1,1);
plot_predictions(target,BLUE,ax);
ax=subplot(2,1,2);
plot_predictions(preds1,BLUE,ax);
[tp,fp,fn]=loacl_calc_hits(target, preds1);
sgtitle(sprintf("Scenario 1 \nTP: %i FP: %i FN: %i",tp,fp,fn))
set(gcf,"WindowStyle","docked")

% Create test predictions 2
preds2 = preds1;

% Add false predictions after at intervals 5-6 9-10 25-26
fpreds2 = [6,9,25;8,10,26]';
s=5*60;
for i = 1:height(fpreds2)
    start=fpreds2(i,1)*s;
    stop=fpreds2(i,2)*s;
    idx=start:stop;
    preds2(idx)=1;
end

figure(2); clf;
ax=subplot(2,1,1);
plot_predictions(target,BLUE,ax);
ax=subplot(2,1,2);
plot_predictions(preds2,BLUE,ax);
[tp,fp,fn]=loacl_calc_hits(target, preds2);
sgtitle(sprintf("Scenario 2 \nTP: %i FP: %i FN: %i",tp,fp,fn))
set(gcf,"WindowStyle","docked")

% Create predictions 3
% Shift all predictions by half
preds3 = preds;
for i = 1:numel(tStop)
    interVal = tStart(i):tStop(i);
    interVal = interVal - (tStop(i)-tStart(i)+1);
    preds3(interVal) = 1;
end
figure(3); clf;
ax=subplot(2,1,1);
plot_predictions(target,BLUE,ax);
ax=subplot(2,1,2);
plot_predictions(preds3,BLUE,ax);
[tp,fp,fn]=loacl_calc_hits(target, preds3);
sgtitle(sprintf("Scenario 3 \nTP: %i FP: %i FN: %i",tp,fp,fn))
set(gcf,"WindowStyle","docked")
linkaxes(findall(gcf,"Type","Axes"))


%% 

target = testUsleep{testUsleep.id=="tG6i","yTrue"}{1};
[tStart, tStop, ~] = get_first_and_last(target,1);

mdl = "uSleep";
preds = test{test.model==mdl & test.id=="tG6i","yHat"}{1};

figure(5); clf;
ax=subplot(2,1,1);
plot_predictions(target,BLUE,ax);
ax=subplot(2,1,2);
plot_predictions(preds,BLUE,ax);
[tp,fp,fn]=loacl_calc_hits(target, preds);
sgtitle(sprintf("Scenario %s \nTP: %i FP: %i FN: %i",mdl,tp,fp,fn))
set(gcf,"WindowStyle","docked")
linkaxes(findall(gcf,"Type","Axes"))



%%

function [TP, FP, FN] = loacl_calc_hits(target, pred)

[tStart, tStop, ~] = get_first_and_last(target,1);
[pStart, pStop, ~] = get_first_and_last(pred, 1);


tScored = zeros(1,length(tStart));
pScored = zeros(1,length(tStop));

for i = 1:numel(pStart)
    
    match = ((tStop >= pStart(i)) & (pStop(i) >= tStart));
    
    if any(match)
        tScored(match) = 1;
        pScored(i) = 1;
    else
        pScored(i) = 0;
    end
end

TP = sum(pScored==1);
FP = sum(pScored==0);
FN = sum(tScored==0);

end
