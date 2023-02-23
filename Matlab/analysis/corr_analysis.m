%% Correlation analysis between microsleep feautures and sleep latency (SL)

setup



%% Collect per subject predictions and targets

% Run main and check_overfitting first
% new_main; check_overfitting;

all_mUSleep = [testMuSleep; trainMuSleep];
all_mUSleep.probs=cellfun(@(x) x(:,2), all_mUSleep.probs, 'UniformOutput', false);
all_USleep = [testUsleep; trainUSleep];
all_SSL = [testSSL; trainSSL];
all_lstm = [testLSTM; trainLSTM];
all_rf = [testRF; trainRF];
all_svm = [testSVM; trainSVM];
all_cnn = [testMalafeev; trainMalafeev];
all_expert = all_rf;
all_expert.yHat = all_rf.yTrue;

data = {all_mUSleep;
    all_USleep;
    all_SSL;
    all_lstm;
    all_rf;
    all_svm;
    all_cnn;
    all_expert};

models = [mdls; "Human"];
tbl = readtable("Skorucak2020\2023_data_raw_AASM.xlsx");

slMethod = @(x) get_aasm_sl(tbl, x, "aasm_sl_sec", 0);
% slMethod = @(x) get_moving_sleep_latency(x, 15);


%% Compute correlation stats with all predictions
stats = table;

tbl = readtable("Skorucak2020\2023_data_raw_AASM.xlsx");
for i = 1:length(data)
    tmpData = data{i};
    tmpStats = table;
    for j = 1:height(tmpData)
        t = tmpData(j,:);
        entry = get_corr_stats(t, params, slMethod);
        entry.train = ismember(t.id, params.split.train);
        entry.id = t.id;
        tmpStats = [tmpStats; struct2table(entry)];
    end
    tmpStats.model = repelem(models(i),height(tmpStats),1);
    stats = [stats; tmpStats];
end
writetable(stats,"../R/all_data.csv");

%% Comput correlation stats with limited predictions
limits = [3, 15]; % limit to 3-15 seconds
stats = table;
for i = 1:length(data)
    tmpData = data{i};
    tmpStats = table;
    for j = 1:height(tmpData)
        t = tmpData(j,:);
        entry = get_corr_stats(t, params, slMethod, [3, 15]);
        entry.id = t.id;
        entry.train = ismember(t.id, params.split.train);
        tmpStats = [tmpStats; struct2table(entry)];
    end
    tmpStats.model = repelem(models(i),height(tmpStats),1);
    stats = [stats; tmpStats];
end
writetable(stats,"../R/limited_data.csv");


%% Collect the sleep onset and probabilties

slMethod = @(x) get_aasm_sl(tbl, x, "aasm_sl_sec", 15);
% slMethod = @(x) get_moving_sleep_latency(x);
probStats = table;
recLen = 2400;
compress = @(x) mat2cell(x,size(x,1),size(x,2));
for i = 1:length(data)
    tmpData = data{i};
    tmpStats = table;
    if ~any(contains(tmpData.Properties.VariableNames, "probs"));
        continue
    end

    for j = 1:height(tmpData)
        t = tmpData(j,:);

        yTrue = my_cell2mat(t.yTrue);
        entry = struct;
        entry.id = t.id;
        entry.train = ismember(t.id, params.split.train);

        try
            entry.sl = slMethod(yTrue);
        catch
            entry.sl = slMethod(t.id);
        end
        probs = my_cell2mat(t.probs);
        if size(probs,1) > size(probs,2); probs = probs'; end

        % Limit probs to sl
        probs = probs(1:entry.sl);

        pad = recLen - length(probs);
        padProbs = [nan(1,pad),probs];
        entry.probs = compress(padProbs);
        entry.model = models(i);
        tmpStats = [tmpStats; struct2table(entry)];
    end
    tmpStats.model = repelem(models(i),height(tmpStats),1);
    probStats = [probStats; tmpStats];
end

%%
ps = probStats;

preMin = @(in, far, close) cellfun(@(x) mean(x(length(x)-(far*60):length(x)-(close*60)), "omitnan"), in);

[idx, m] = findgroups(ps(:,"model"));
mins = [5, 0;
        10, 5;
        20, 10];
for i = 1:height(mins)
    far = mins(i,1);
    close = mins(i,2);
    pre=splitapply(@(x){preMin(x, far, close)}, ps.probs, idx);
    tblStr = sprintf("prob_%i_to_%i_pre_sleep_onset", far, close);
    ps.(tblStr)= vertcat(pre{:});
end

ps=removevars(ps,"probs");
writetable(ps,"..\R\intervaled_probabilites.csv")


%%
slMethod = @(x) get_aasm_sl(tbl, x, "aasm_sl_sec", 15);

probStatsAASM = table;
recLen = 2400;
compress = @(x) mat2cell(x,size(x,1),size(x,2));
for i = 1:length(data)
    tmpData = data{i};
    tmpStats = table;
    if ~any(contains(tmpData.Properties.VariableNames, "probs"));
        continue
    end

    for j = 1:height(tmpData)
        t = tmpData(j,:);

        yTrue = my_cell2mat(t.yTrue);
        entry = struct;
        entry.id = t.id;
        entry.train = ismember(t.id, params.split.train);

        try
            entry.sl = slMethod(yTrue);
        catch
            entry.sl = slMethod(t.id);
        end
        probs = my_cell2mat(t.probs);
        if size(probs,1) > size(probs,2); probs = probs'; end

        % Limit probs to sl
        probs = probs(1:entry.sl);

        pad = recLen - length(probs);
        padProbs = [nan(1,pad),probs];
        entry.probs = compress(padProbs);
        entry.model = models(i);
        tmpStats = [tmpStats; struct2table(entry)];
    end
    tmpStats.model = repelem(models(i),height(tmpStats),1);
    probStatsAASM = [probStatsAASM; tmpStats];
end

slMethod = @(x) get_moving_sleep_latency(x);

probStatsWin = table;
recLen = 2400;
compress = @(x) mat2cell(x,size(x,1),size(x,2));
for i = 1:length(data)
    tmpData = data{i};
    tmpStats = table;
    if ~any(contains(tmpData.Properties.VariableNames, "probs"));
        continue
    end

    for j = 1:height(tmpData)
        t = tmpData(j,:);

        yTrue = my_cell2mat(t.yTrue);
        entry = struct;
        entry.id = t.id;
        entry.train = ismember(t.id, params.split.train);

        try
            entry.sl = slMethod(yTrue);
        catch
            entry.sl = slMethod(t.id);
        end
        probs = my_cell2mat(t.probs);
        if size(probs,1) > size(probs,2); probs = probs'; end

        % Limit probs to sl
        probs = probs(1:entry.sl);

        pad = recLen - length(probs);
        padProbs = [nan(1,pad),probs];
        entry.probs = compress(padProbs);
        entry.model = models(i);
        tmpStats = [tmpStats; struct2table(entry)];
    end
    tmpStats.model = repelem(models(i),height(tmpStats),1);
    probStatsWin = [probStatsWin; tmpStats];
end

%%
probModel = unique(probStatsAASM.model);
for mi = 1:numel(probModel)
    mdl = probModel(mi);
    figure(42); clf; hold on;
    for j = 1:2
        if j == 1;
            tmp = probStatsAASM;
        else
            tmp = probStatsWin;
        end

        m = probStats(probStats.model == mdl,:);
        probs = cell2mat(m.probs);

        allNans = all(isnan(probs));
        probs = probs(:,~allNans);
        ttime = time(1:length(probs));
        revTime = ttime - max(ttime);

        % Average probs
        avgProbs = mean(probs,1,"omitnan");

        % Sem across record probs
        semProbs = std(probs,[],1,"omitnan") ./ sqrt(sum(~isnan(probs)));

        % Moving average probs
        win = 30;
        movProbs = movmean(avgProbs, win, "omitnan", "Endpoints","shrink");

        movSemProbs = movmean(semProbs, win, "omitnan", "Endpoints","shrink");

        subplot(1,2,j); hold on;
        % Plot sem patch
        % y = movProbs; s = movSemProbs;
        y = avgProbs; s = semProbs;
        p=patch(gca,[revTime fliplr(revTime)], [y-semProbs  fliplr(y+semProbs)], ...
            colors.BLUE, "EdgeColor","none", "FaceColor", colors.BLUE);
        p.EdgeColor = "none";
        p.FaceAlpha = 0.2;

        % Plot all point average probabilties
        plot(revTime, avgProbs, ...
            'LineWidth',1, 'LineStyle',':', "Color",colors.BLUE);

        % Plot moving average
        plot(revTime, movProbs, '-', ...
            'LineWidth',4 ,'Color',colors.BLUE);
        xlim([min(revTime) max(revTime)+1])
        
        xline(0, "--", "LineWidth",2)

        legend(["SEM","Mean probabilites","Smoothed mean probabilites","Sleep Onset"])
       
        xlabel("Time before sleep onset [min]")
        ylabel("Mean Probability of MS")
        if j==1
            title("SL defined by: AASM-SL minus 15 seconds", ...
                "FontWeight","normal")
        else
            title("SL defined by: First moving window > 15 sec sleep", ...
                "FontWeight","normal")
        end
        sgtitle(sprintf("Probabilites from: %s\n",mdl), "FontWeight","bold")
    box on
    end
    set(findall(gcf,"-property","FontSize"),"FontSize",12)
    set(gcf,"WindowState","fullscreen")
    pause
%     figFile = fullfile("C:\\Users\\Sindri\\Desktop\\EM\\",sprintf("%s_probability_vs_SL.png", mdl));
%     exportgraphics(gcf, figFile, "Resolution",300);
end



%% Plot to see labels and durations against trial termination
% figure(1); clf;
% time = linspace(0,40,40*60);
% tmp = all_expert;
% saveOn = 0;
% for i = 1:height(tmp)
% %     i = randi(76,1,1);
%
%     sl = nan;
%
%     t = tmp(i,:);
%     yt=cell2mat(t.yTrue);
%     ax1=subplot(2,1,1); hold on;
%     plot_predictions(yt,time,colors.BLUE,1,ax1);
%     subtitle(ax1,t.id);
%     xlim([min(time),max(time)])
%     lineHdl = xline(length(yt)/60,"r:","LineWidth", 5);
%     box(ax1,"on");
%     msHdl = findall(ax1,"Type","Patch");
%
%     try
%        [sl, ms] = yMethod(yt);
%     catch
%        try
%            sl = yMethod(yt);
%        catch
%            sl = yMethod(t.id);
%        end
%     end
%
%     sl = sl / 60;
%
%
%     slHdl = xline(ax1,sl,"g--","LineWidth", 5, ...
%         "DisplayName","Sleep Latency");
%     lHdl = [lineHdl, slHdl];
%
%     numEpochs = length(yt) / 30;
%     hold(ax1,"on");
%     for j=1:numEpochs
%         e=j*30 / 60;
%         xline(ax1,e,'-',"color",[90,90,90]/255,"LineWidth",1);
%     end
%
%     if isempty(msHdl)
%         l=legend(ax1, lHdl, ["End of recording", "Sleep Latency"]);
%     else
%         l=legend(ax1,[msHdl(1), lHdl], ["MS","End of recording","Sleep Latency"]);
%     end
%     l.Location = "southwest";
%     l.AutoUpdate = "on";
%
%
%     [first, last, singles] = get_first_and_last(yt,1,false);
%     durs = ((last - first)+1)*params.secPerLabel;
%     durs = [durs, ones(size(singles))*params.secPerLabel];
%
%     ax2=subplot(2,1,2); hold on;
%     box(ax2, "on");
%     grid(ax2,"on");
%     try
%     plot(ax2,time(1:length(ms)),ms);
%     yline(ax2, 15, '--', "color",colors.BLUE);
%     catch
%     end
%     linkaxes([ax1,ax2],"x")
%     if any(durs)
%         dIdx = [(last+first) / 2, singles] / 60;
%         [dIdx, ii] = sort(dIdx);
%         durs = durs(ii);
%         scatter(ax2, dIdx, durs, 30, "o", "filled");
%         yline(ax2, 30, '--', "Color",colors.ORANGE);
%         ylim([0, (max(ax2.YLim(end),60))])
%         xlim([min(time),max(time)])
%         box(ax2, "on");
%         grid(ax2,"on")
%
%     end
%     ylabel(ax2,"Duration (s)")
%     xlabel(ax2,"Time (min)")
%
%
%     if saveOn
%         set(findall(gcf,"-property","FontSize"),"FontSize",12)
%         set(gcf,"WindowState","fullscreen")
%         figFile = fullfile("C:\\Users\\Sindri\\Desktop\\Oliver\\",sprintf("%s.png", t.id));
%         exportgraphics(gcf, figFile, "Resolution",300);
%     end
%     pause
%     clf;
% end

%%
% t = tmp(i-1,:);
% yt = my_cell2mat(t.yTrue);
% plot_predictions(yt,time,colors.BLUE)
% sl = get_sleep_latency(yt)

%%

function ax = plot_probs_to_sl(m, params, ax)

if nargin < 4; ax = gca; end
time = linspace(0,40,2400);
hold(ax,"on")

m = probStats(probStats.model == mdl,:);
probs = cell2mat(m.probs);

allNans = all(isnan(probs));
probs = probs(:,~allNans);
ttime = time(1:length(probs));
revTime = ttime - max(ttime);

% Average probs
avgProbs = mean(probs,1,"omitnan");

% Sem across record probs
semProbs = std(probs,[],1,"omitnan") ./ sqrt(sum(~isnan(probs)));

% Moving average probs
win = 30;
movProbs = movmean(avgProbs, win, "omitnan", "Endpoints","shrink");

movSemProbs = movmean(semProbs, win, "omitnan", "Endpoints","shrink");

% Plot sem patch
% y = movProbs; s = movSemProbs;
y = avgProbs; s = semProbs;


p=patch(ax,[revTime fliplr(revTime)], [y-semProbs  fliplr(y+semProbs)], ...
    colors.BLUE, "EdgeColor","none", "FaceColor", colors.BLUE);
p.EdgeColor = "none";
p.FaceAlpha = 0.2;

% Plot all point average probabilties
plot(ax,revTime, avgProbs, ...
    'LineWidth',1, 'LineStyle',':', "Color",colors.BLUE);

% Plot moving average
plot(ax,revTime, movProbs, '-', ...
    'LineWidth',4 ,'Color',colors.BLUE);
xlim(ax,[min(revTime) max(revTime)+1])

xline(ax,0, "--", "LineWidth",2)

legend(ax,["SEM","Mean probabilites","Smoothed mean probabilites","Sleep Onset"])

xlabel(ax,"Time before sleep onset [min]")
ylabel(ax,"Mean Probability of MS")
box on
hold(ax,"off")
end

function out = get_corr_stats(rec, params, yMethod, limits)

if nargin < 4; limits = []; end

% Extract yHat, yTrue and yProbs
fn = fieldnames(rec);
yHat = my_cell2mat(rec.yHat);
yTrue = my_cell2mat(rec.yTrue);
isProb = any(contains(fn,"probs"));
if isProb; yProbs = my_cell2mat(rec.probs); end

if ~isempty(limits)
    % Apply time criteria to yHat
    minLim = limits(1); maxLim = limits(end);
    if minLim > 0; skipSingles = false; else; skipSingles = true; end
    yHat = remove_invalid_labels(yHat, minLim, maxLim, ...
        params.secPerLabel, nan, skipSingles);
end

% Struct to store info
out = struct;

% === Get y-variable (sleep latency) ===
durLimit = 30;

%   Evaluated as first label (yTrue) > 30sec or length of rec
[first, last, ~] = get_first_and_last(yTrue, 1, false);
durs = ((last - first)+1)*params.secPerLabel;
onset = find(durs > durLimit, 1, "first");

try
    out.y = yMethod(yTrue);
catch
    out.y = yMethod(rec.id);
end

numMin = out.y / 60;
yLim = min(length(yTrue),out.y);

% Set yHat to range of y-variable
yHat = yHat(1:yLim);

% === Compute x-variables ===

% Number of MS events per minute
[first, last, singles] = get_first_and_last(yHat, 1, false);
out.x_countMS = sum(length(first)+length(singles));
out.x_numMS = sum(length(first) + length(singles)) / numMin;

% Cumulative duration of MS
durs = get_durations(yHat, params.secPerLabel);
out.x_cumDurMS = sum(durs);

% Median MS duration
out.x_medDurMS = max(median(durs),0);

% Mean MS duration
out.x_meanDurMS = max(mean(durs), 0);

% Inter MS-interval
first=sort([first, singles+1e-4]);
last=sort([last, singles-1e-4]);
interDurs = round((first(2:end) - last(1:end-1))*params.labelsPerSec);

%   Mean
out.x_interDurMeanMS = max(mean(interDurs),0);
%   Median
out.x_interDurMedianMS = max(median(interDurs),0);

% Probability features
if isProb
    yProbs = yProbs(1:yLim);
    % Cumulative probability
    out.x_cumProbMS = sum(yProbs) / length(yProbs);
    % Gradient of cumuluative sum
    out.x_probEntropyMS = wentropy(yProbs,"shannon");

    % Get time based probability features
    % Mean probability within these time ranges
%     pre5 = idx_pre_sleep_onset(yProbs, 5);
%     pre10 = idx_pre_sleep_onset(yProbs, 5);
%     pre20 = idx_pre_sleep_onset(yProbs, 5);
%     
else
    out.x_cumProbMS = nan;
    out.x_probEntropyMS = nan;
end
end

function idx = idx_pre_sleep_onset(sig, min)

    idx = length(sig) - (min*60);
    
    if idx < 0
        idx = nan;
    elseif idx == 0
        idx = 1;
    end
       
end

function sleepLatency = get_per_epoch_sleep_latency(y, sampPerEpoch)

if nargin < 2; sampPerEpoch = 30; end

sleepThres = sampPerEpoch * 0.5;


% Reshape into n x 30, where n is equal to length(y) / 30
n = length(y) / sampPerEpoch;
%     idx = (1:length(y))';

if size(y,1) > size(y,2)
    epochs = reshape(y, [sampPerEpoch, n])';
    %         ei = reshape(idx, [sampPerEpoch, n])';
else
    epochs = reshape(y', [n, sampPerEpoch])';
    %         ei = reshape(idx', [n, sampPerEpoch])'
end

perEpoch = sum(epochs==1, 2);

onsetEpoch = find(perEpoch >= sleepThres, 1, "first");

if any(onsetEpoch)
    sleepLatency = (onsetEpoch-1) * sampPerEpoch;
else
    sleepLatency = length(y);
end

end

function [sleepLatency, winSleep] = get_moving_sleep_latency(y, lim)
if nargin < 2; lim = 15; end
sleep = y == 1;
epoch = 30;
winSleep = movsum(sleep, epoch, "Endpoints","shrink");

sleepLatency = find(winSleep >= 15, 1, "first");
if isempty(sleepLatency)
    sleepLatency = length(y);
else
    sleepLatency = sleepLatency - epoch;
end

end

function out = get_aasm_sl(tbl, id, var, reduce)
if nargin < 4; reduce = 0; end
out = double(tbl{lower(tbl.id) == lower(id), var});
out = out - reduce;
end


function out=my_cell2mat(x)
try
    out = cell2mat(x);
catch ME
    if strcmp(ME.identifier, 'MATLAB:catenate:dimensionMismatch')
        out = cell2mat(x');
    else
        throw(ME)
    end
end
end

