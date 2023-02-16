% Main script

setup

% 0. Set the parameters of the main script
params = struct;
params.outFolder = "comparison_2_train_test";
params.filePaths = strcat("data/", ls("data/*.mat"));
params.fileIds = params.filePaths.extractAfter("data/").extractBefore(".mat");
params.seed = load("seed.mat");
params.split = parse_json("..\splits\skorucack_splits.json");
params.masterFile = fullfile(params.outFolder, "master.mat");

% Label specifics
params.posClassValue = 1;
params.negClassValue = -1;
params.trainTargets = "biWake_vs_biMSE";
params.testTargets = "newTarget";

% Time domain values
params.secPerLabel   = 0.2;
params.labelsPerSec  = 1 / params.secPerLabel;
params.windowSizeSec = 0.2;
params.windowStepSec = 0.2;
params.downsampleFcn = @(x) my_median(x); % down sampling technique
params.minDurationSec = 3;
params.maxDurationSec = inf;
params.replaceInvalidPredictions = nan;
params.skipSingles = false;

% Features
params.zeroPad = zeros(1, 0.4*200); % 40 samples account for the first and last two samples lost with the moving window
params.numFeatures = 14;
params.pburgWindowSize = 1;

% Model specifics
params.retrain = ["RF","SVM","LSTM"];
params.windowSizeLSTM = 45;

% U-sleep specifics
usleep = struct;
usleep.splitValue = 0.5;
usleep.hz = [32, 16, 8, 1];
usleep.thresholds = (0.025:0.025:1)';
usleep.outfile = fullfile(params.outFolder, "uSleep.mat");
usleep.evalMetric = "kappa";
params.usleep = usleep;



%% 0. Generate the folder of the analysis if it doesn't exist
% Store the params in the folder and all of the the models
init_project(params);


%% 1. Generate master table with labels and feature from Skorucak 2020

master = init_master(params);

%% 1.a Apply different duration criteria to MS

% Get the labels per recordings
[idx, rec] = findgroups(master.id);
recLabels = splitapply(@(x) {x}, master{:,params.trainTargets}, idx);
orgDurs = cellfun(@(x) get_durations(x, params.secPerLabel), recLabels, ...
                  "UniformOutput",false);
% orgDurs = horzcat(orgDurs{:});
% orgDurStats = calc_stats(orgDurs);
% orgDurStats.withinLimit = sum(orgDurs>=3 & orgDurs<=15) / length(orgDurs);


% Set labels less than 3 seconds to negative class (wake)
newLabels1 = cellfun(@(x) remove_invalid_labels(x, ...
                                                params.minDurationSec, ...
                                                inf, ...
                                                params.secPerLabel, ...
                                                params.negClassValue), ...
                                                recLabels, ...
                                                "UniformOutput", false);
% Set labels more than 15 seconds to nan (invalid)
newLabels2 = cellfun(@(x) remove_invalid_labels(x, ...
                                                0, ...
                                                15, ...
                                                params.secPerLabel, ...
                                                nan), ...
                                                newLabels1, ...
                                                "UniformOutput", false);                                        

% Calculate statistics of new labels
newDurs = cellfun(@(x) get_durations(x, params.secPerLabel), newLabels2, ...
                  "UniformOutput",false);
newDurStats = calc_stats(cell2mat(newDurs'));

master.newTarget = vertcat(newLabels2{:});

trainDurs = cell2mat(orgDurs(ismember(rec,params.split.train))');
testDurs  = cell2mat(newDurs(ismember(rec,params.split.test))');

trainDurStats = calc_stats(trainDurs)
testDurStats = calc_stats(testDurs)

figure(100); clf;
hold on
histogram(trainDurs,50,"Normalization","probability","FaceAlpha",0.5);
histogram(testDurs,50,"Normalization","probability","FaceAlpha",0.5);
% xticks(0:1:15)
xlabel("Duration [sec]")
ylabel("Density")
legend(["Training set","Test set"])

% How many MSE vs Wake?
trainIdx = ismember(master.id, params.split.train);
trainFractions = struct;
trainFractions.MSE = sum(master{trainIdx,params.testTargets} == 1) / sum(trainIdx);
trainFractions.Wake = sum(master{trainIdx,params.testTargets} == -1) / sum(trainIdx);
trainFractions.nan = 1-(trainFractions.MSE+trainFractions.Wake);
trainFractions

testIdx = ismember(master.id, params.split.test);
testFractions = struct;
testFractions.MSE = sum(master{testIdx,params.testTargets} == 1) / sum(testIdx);
testFractions.Wake = sum(master{testIdx,params.testTargets} == -1) / sum(testIdx);
testFractions.nan = 1-(testFractions.MSE+testFractions.Wake);
testFractions

%% 2. Train and evaluate RF model

[RF, testRF] = train_and_test_model("RF", params, master);
[RF_overallMetrics, RF_perRecMetrics] = eval_test_by(testRF, "sample", params);

%% 3. Train and evaluate SVM model

[SVM, testSVM] = train_and_test_model("SVM", params, master);
[SVM_overallMetrics, SVM_perRecMetrics] = eval_test_by(testSVM, "sample", params);

%% 4. Train and eval LSTM model

[LSTM, testLSTM] = train_and_test_lstm(params, master);
[LSTM_overallMetrics, LSTM_perRecMetrics] = eval_test_by(testLSTM, "sample", params);

%% 5. Evaluate optimal U-Sleep processing model and test on optimal model
uSleepPredictions = process_usleep_predictions(params, master);

[cvUsleep, optHz, optMethod] = cv_usleep(params, uSleepPredictions, master);

[optUsleep, testUsleep, optThres] = train_and_test_usleep(uSleepPredictions, optHz, optMethod, params);

[uSleep_overallMetrics, uSleep_perRecMetrics] = eval_test_by(testUsleep, "sample", params);


%% 5.1 CV analysis

cvStats = grpstats(cvUsleep,["method","hz"],["mean","sem"],...
    "DataVars","kappa");
cvStats.Properties.RowNames = {};
hz = unique(cvStats.hz);
m = unique(cvStats.method);
offset = linspace(-1,1,length(m));
figure(1);clf;hold on
cc=lines(length(m));
x=linspace(2,20,4);
for i = 1:numel(hz)
    lhdl=[];
    for j = 1:numel(m) 
        tmp = cvStats((cvStats.hz == hz(i) & cvStats.method == m(j)),:);
        if ~isempty(tmp) 
            p=plot(x(i)+offset(j), tmp.mean_kappa, ...
                     "Marker","o","MarkerFaceColor",cc(j,:),"MarkerEdgeColor",cc(j,:));
            errorbar(x(i)+offset(j), tmp.mean_kappa, tmp.sem_kappa, ...
            "Marker","none","Color",cc(j,:));
        end
    lhdl = [lhdl, p];
    end
end
xlim([0, 22])
xticks(x);
xticklabels(hz);
xlabel("Hz")
ylabel("Cohen's Kappa (+/- SEM)")
set(findall(gca,"Type","Line"),"MarkerSize",8)
set(findall(gca,"Type","ErrorBar"),"LineWidth",2)
legend(lhdl, m, "NumColumns",3,"Location","southeast")
grid on
box on
set(findall(gcf,'-property','FontSize'),'FontSize',14)
optCV=cvStats(cvStats.hz==optHz & cvStats.method==optMethod,:);
title(sprintf("Optimal model: %s @ %i Hz = %.2f", ...
                              optMethod, optHz, optCV.mean_kappa))

%% 5.2 Threshold analysis
thresData = uSleepPredictions(uSleepPredictions.hz == optHz & ...
                              ismember(uSleepPredictions.id,params.split.train), ...
                              ["testTargets",optMethod]);
thresPreds = num2cell(cell2mat(thresData{:,optMethod}'),2);
thresPerformance = cellfun(@(x) calc_performance_metrics(cell2mat(thresData.testTargets'), x, params), ...
                            thresPreds);

thresPerformance = struct2table(thresPerformance);
thresPerformance.threshold = params.usleep.thresholds';

figure(2); clf;
subplot(1,2,1); hold on;
plot(thresPerformance.sensitivity, thresPerformance.precision, ...
    '-o', "LineWidth",1,"MarkerFaceColor",colors.BLUE)
optIdx = thresPerformance.threshold==optThres;
plot(thresPerformance.sensitivity(optIdx), thresPerformance.precision(optIdx), ...
     "Marker","*","Color",colors.YELLOW,"MarkerSize",12,"LineWidth",2)
grid on
box on
xlim([-0.01, 1])
ylim([0, 1])
xlabel("Recall")
ylabel("Precision")

subplot(1,2,2); hold on;
plot(thresPerformance.threshold, thresPerformance.kappa, ...
    '-o', "LineWidth",1, "MarkerFaceColor",colors.BLUE)
o=plot(optThres, thresPerformance.kappa(optIdx), ...
     "Marker","*","Color",colors.YELLOW,"MarkerSize",12,"LineWidth",2);
grid on
xlim([0, 1])
ylim([0, 1])
box on
legend(o,"Optimal Threshold");
xlabel("Thresholds")
ylabel("Cohen's Kappa")
set(findall(gcf,"-property","Fontsize"),"FontSize",12)

%% 6. Summarize the results


mdls = ["uSleep";
          "LSTM";
          "RF";
          "SVM"];

% Overall      
overallSampleTable = struct2table([uSleep_overallMetrics;
                            LSTM_overallMetrics;
                            RF_overallMetrics;
                            SVM_overallMetrics]);
                        
overallSampleTable.model = mdls;
overallSampleTable = overallSampleTable(:,["model",overallSampleTable.Properties.VariableNames(1:end-1)]);


% Per rec
perRecSample = who('*_perRecMetrics');
perRecSampleTable = table;
for i = 1:numel(perRecSample)
    var = perRecSample{i};
    [m, s] = summarize_perRecMetrics(eval(var), "sample");
    info = cellfun(@(m,s) sprintf("%.2f +/- %.2f",m,s), struct2cell(m),struct2cell(s));
    entry = cell2struct(cellstr(info),fieldnames(m));
    entry.model = string(var).extractBefore("_");
    perRecSampleTable = [perRecSampleTable; struct2table(entry)];
end
perRecSampleTable = perRecSampleTable([4,1:3],["model",perRecSampleTable.Properties.VariableNames(1:end-1)]);

disp(overallSampleTable)
perRecSampleTable.Variables

% Save
results = struct;
results.overall = overallSampleTable;
results.perRec = perRecSampleTable;
resultsFile = fullfile(params.outFolder,"sampleResults.mat");
save(resultsFile,"results");

%%
write_results_table(overallSampleTable,perRecSampleTable,"sample",params)

%% 7. Summarize per event metrics
[RF_eventOverallMetrics, RF_eventPerRecMetrics] = eval_test_by(testRF, "event", params);
[SVM_eventOverallMetrics, SVM_eventPerRecMetrics] = eval_test_by(testSVM, "event", params);
[LSTM_eventOverallMetrics, LSTM_eventPerRecMetrics] = eval_test_by(testLSTM, "event", params);
[uSleep_eventOverallMetrics, uSleep_eventPerRecMetrics] = eval_test_by(testUsleep, "event", params);

% Overall      
overallEventTable = struct2table([uSleep_eventOverallMetrics;
                                  LSTM_eventOverallMetrics;
                                  RF_eventOverallMetrics;
                                  SVM_eventOverallMetrics]);
                        
overallEventTable.model = mdls;
overallEventTable = overallEventTable(:,["model",overallEventTable.Properties.VariableNames(1:end-1)]);

% Per rec
perRecEvent = who('*_eventPerRecMetrics');
perRecEventTable = table;
for i = 1:numel(perRecEvent)
    var = perRecEvent{i};
    [m, s] = summarize_perRecMetrics(eval(var), "event");
    info = cellfun(@(m,s) sprintf("%.2f +/- %.2f",m,s), struct2cell(m),struct2cell(s));
    entry = cell2struct(cellstr(info),fieldnames(m));
    entry.model = string(var).extractBefore("_");
    perRecEventTable = [perRecEventTable; struct2table(entry)];
end
perRecEventTable = perRecEventTable([4,1:3],["model",perRecEventTable.Properties.VariableNames(1:end-1)]);

disp(overallEventTable)
perRecEventTable.Variables

% Save
results = struct;
results.overall = overallEventTable;
results.perRec = perRecEventTable;
resultsFile = fullfile(params.outFolder,"eventReults.mat");
save(resultsFile,"results");

%%
write_results_table(overallEventTable,perRecEventTable,"event",params)

%% 8. Make informative plots of the test data

% Transform usleep test to match
transUsleepTest = table;
transUsleepTest.id = testUsleep.id;
transUsleepTest.yTrue = cellfun(@(x) num2cell(x',1), testUsleep.yTrue);
transUsleepTest.yHat = cellfun(@(x) num2cell(x',1), testUsleep.yHat);

% Join test results into common table
test = [transUsleepTest;
    testLSTM;
    testRF;
    testSVM];

test.model = repelem(mdls,height(transUsleepTest),1);

uniqueIds = unique(test.id);

saveOn = 0;

for i = 1:numel(uniqueIds)

    tmpId = uniqueIds(i);
    tmp = test(test.id==tmpId,:);
    
    % Per sample
    tmpStats = uSleep_perRecMetrics(uSleep_perRecMetrics.id==tmpId,1:end-1);
    info = cellfun(@(var,value) sprintf("%s=%.2f", var, value), ...
        tmpStats.Properties.VariableNames, table2cell(tmpStats));
    sampleInfo = strjoin(info,"        ");

    % Per event 
    tmpStats = uSleep_eventPerRecMetrics(uSleep_eventPerRecMetrics.id==tmpId,1:end-1);
    info = cellfun(@(var,value) sprintf("%s=%.2f", var, value), ...
        tmpStats.Properties.VariableNames, table2cell(tmpStats));
    eventInfo = strjoin(info,"        ");

    % Init figure
    figure(1); clf;
    yhdl = @(ax, x) ylabel(ax, x, "Rotation", 0, "HorizontalAlignment","right");
    yt = @(ax) set(ax, "YTick", [], "YTickLabels",[]);

    subRows = 2+length(unique(test.model));
    subCols = 1;

    timePreds = linspace(0, 40, 40*60*params.labelsPerSec);

    % Plot the target labels
    tmpTargets = tmp.yTrue{1,:};
    ax1 = subplot(subRows, subCols, 1); hold on;
    plot_predictions(tmpTargets, timePreds, colors.BLUE, 1, ax1);
    lgdHdl =[]; lgdName = [];
    p=findall(ax1.Children,"Type","Patch");
    if ~isempty(p); lgdHdl = [p(1)]; lgdName = ["MS"]; end
    plot_predictions(isnan(tmpTargets), timePreds, [220,220,220]/256, 0.8, ax1);
    p2=findall(ax1.Children,"Type","Patch");
    if length(p2) > length(p); lgdHdl = [lgdHdl; p2(1)]; lgdName = [lgdName; "InvalidMS"]; end
    if ~isempty(lgdHdl); legend(lgdHdl, lgdName, "Location","northwest"); end
    xticklabels(ax1,[]);
    yhdl(ax1,"Target")
    yt(ax1)

    % Plot Usleep probabilities
    probFile = sprintf("predictions//%i_hz//%s.mat",optHz,tmpId);
    probs = get_probs(probFile);
    ax2 = subplot(subRows, subCols, 2);
    timeProbs = linspace(0, 40, 40*60*optHz);
    plot_probs(probs, timeProbs, ax2);
    yline(ax2, optThres, "LineWidth",2);
    xticks(ax2,[]);
    xticklabels(ax2,[]);
    yticks(ax2,[0,1])
    % legend(ax2,"off")
    yhdl(ax2, "P(Sleep)")

    for mIdx = 1:numel(mdls)
        % Get tmp model test results
        tmpModel = mdls(mIdx);
        tmpResults = tmp(tmp.model==tmpModel,:);

        % Plot Usleep predictions
        tmpPreds = cell2mat(tmpResults.yHat);
        axMdl = subplot(subRows,subCols,2+mIdx);
        plot_predictions(tmpPreds, timePreds, colors.BLUE, 1, axMdl);
        if mIdx < numel(mdls)
            xticklabels(axMdl,[]);
        else
            xlabel("Time [min]")
        end
        yhdl(axMdl, tmpModel)
        yt(axMdl);
    end
    

    % Link all axes
    axs = findall(gcf,"Type","axes");
    linkaxes(axs)
    set(findall(gcf,"-property","FontSize"),"FontSize",12)
    set(gcf,"WindowState","fullscreen")
    
    for j = 1:numel(axs)
        box(axs(j), "on")
    end
    sgtitle(sprintf("%s\nSample: %s\nEvent: %s",tmpId,sampleInfo,eventInfo));
    
    %Save figure
    if saveOn
        figFile = fullfile(params.outFolder,"figures",sprintf("%s.png",tmpId));
        exportgraphics(gcf, figFile, "Resolution",300);
    end
    

end


