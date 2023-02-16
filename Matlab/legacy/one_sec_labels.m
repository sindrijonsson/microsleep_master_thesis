% Main script

setup

% 0. Set the parameters of the main script
params = struct;
params.outFolder = "one_sec_labels";
params.filePaths = strcat("data/", ls("data/*.mat"));
params.fileIds = params.filePaths.extractAfter("data/").extractBefore(".mat");
params.seed = load("seed.mat");
params.split = parse_json("..\splits\skorucack_splits.json");
params.masterFile = fullfile(params.outFolder, "master.mat");

% Label specifics
params.posClassValue = 1;
params.negClassValue = -1;
params.trainTargets = "biWake_vs_biMSE";
params.testTargets = "biWake_vs_biMSE";

% Time domain values
params.secPerLabel   = 1;
params.labelsPerSec  = 1 / params.secPerLabel;
params.windowSizeSec = 1;
params.windowStepSec = 1;
params.downsampleFcn = @(x) my_median(x); % down sampling technique
params.minDurationSec = 1;
params.maxDurationSec = inf;
params.skipSingles = true;
params.replaceInvalidPredictions = nan;

% Features
params.zeroPad = zeros(1, 0); % 40 samples account for the first and last two samples lost with the moving window
params.numFeatures = 14;
params.pburgWindowSize = 1;

% Model specifics
params.retrain = [""];%["RF","SVM","LSTM"];
params.windowSizeLSTM = 9;

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
orgDurStats = calc_stats(cell2mat(orgDurs'))

trainDurs = cell2mat(orgDurs(ismember(rec,params.split.train))');
testDurs  = cell2mat(orgDurs(ismember(rec,params.split.test))');

trainDurStats = calc_stats(trainDurs)
testDurStats = calc_stats(testDurs)

figure(100); clf;
hold on
histogram(trainDurs,50,"Normalization","probability","FaceAlpha",0.5);
histogram(testDurs,50,"Normalization","probability","FaceAlpha",0.5);
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

%% 0.1 Apply time criteria?
apply_time_criteria = false;

%% 2. Train and evaluate RF model

[RF, testRF] = train_and_test_model("RF", params, master);
[RF_overallMetrics, RF_perRecMetrics] = eval_test_by(testRF, "sample", params);

if apply_time_criteria
    critRF = testRF;
    short_crit = cellfun(@(x) remove_invalid_labels(x, 3, inf, 1, -1, false), critRF.yTrue, "UniformOutput", false);
    long_crit = cellfun(@(x) remove_invalid_labels(x, 0, 15, 1, nan, true), short_crit, "UniformOutput", false);
    critRF.yTrue = long_crit;
    critRF.yhat = cellfun(@(x) remove_invalid_labels(x, 3, inf, 1, -1, false), critRF.yHat, "UniformOutput",false);
    [RF_overallMetrics, RF_perRecMetrics] = eval_test_by(critRF, "sample", params);
end

%% 3. Train and evaluate SVM model

[SVM, testSVM] = train_and_test_model("SVM", params, master);
[SVM_overallMetrics, SVM_perRecMetrics] = eval_test_by(testSVM, "sample", params);

if apply_time_criteria

critSVM = testSVM;
short_crit = cellfun(@(x) remove_invalid_labels(x, 3, inf, 1, -1, false), critSVM.yTrue, "UniformOutput", false);
long_crit = cellfun(@(x) remove_invalid_labels(x, 0, 15, 1, nan, true), short_crit, "UniformOutput", false);
critSVM.yTrue = long_crit;
critSVM.yhat = cellfun(@(x) remove_invalid_labels(x, 3, inf, 1, -1, false), critSVM.yHat, "UniformOutput",false);
[SVM_overallMetrics, SVM_perRecMetrics] = eval_test_by(critSVM, "sample", params);

end

%% 4. Train and eval LSTM model

[LSTM, testLSTM] = train_and_test_lstm(params, master);
[LSTM_overallMetrics, LSTM_perRecMetrics] = eval_test_by(testLSTM, "sample", params);

if apply_time_criteria

    critLSTM = testLSTM;
    short_crit = cellfun(@(x) remove_invalid_labels(x, 3, inf, 1, -1, false), critLSTM.yTrue, "UniformOutput", false);
    long_crit = cellfun(@(x) remove_invalid_labels(x, 0, 15, 1, nan, true), short_crit, "UniformOutput", false);
    critLSTM.yTrue = long_crit;

    critLSTM.yhat = cellfun(@(x) remove_invalid_labels(x, 3, inf, 1, -1, false), critLSTM.yTrue, "UniformOutput",false);
    [LSTM_overallMetrics, LSTM_perRecMetrics] = eval_test_by(critLSTM, "sample", params);
end

%% 5. Evaluate optimal U-Sleep processing model and test on optimal model
uSleepPredictions = process_usleep_predictions(params, master);

[cvUsleep, optHz, optMethod] = cv_usleep(params, uSleepPredictions, master);

[optUsleep, testUsleep, optThres] = train_and_test_usleep(uSleepPredictions, optHz, optMethod, params);
%%
[uSleep_overallMetrics, uSleep_perRecMetrics] = eval_test_by(testUsleep, "sample", params);

if apply_time_criteria
    
    critUsleep = testUsleep;
    short_crit = cellfun(@(x) remove_invalid_labels(x, 3, inf, 1, -1, false), critUsleep.yTrue, "UniformOutput", false);
    long_crit = cellfun(@(x) remove_invalid_labels(x, 0, 15, 1, nan, true), short_crit, "UniformOutput", false);
    critUsleep.yTrue = long_crit;
    
    critUsleep.yhat = cellfun(@(x) remove_invalid_labels(x, 3, inf, 1, -1, false), critUsleep.yHat, "UniformOutput",false);
    [uSleep_overallMetrics, uSleep_perRecMetrics] = eval_test_by(critUsleep, "sample", params);

end

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

%% 6, Transfer learning mat
% Let's evaluate transfer learning results
TF = load(fullfile(params.outFolder,"transfer_learning_long2.mat"));
TF.id = string(TF.id);
% TF.yHat = TF.yHat';
TF.yTrue = TF.yTrue';
TF.probs = TF.probs';
TF = struct2table(TF);

% Sort according to testUsleep
rows=arrayfun(@(x) find(x==TF.id), testUsleep.id);
TF = TF(rows,:);
TF.yTrue = testUsleep.yTrue;


%tfOptThres = 0.275;
tfOptThres = 0.45848149061203003;


yHat = [];
for i = 1:height(TF)
    tmpHat = double(TF.probs{i}(:,2) > tfOptThres);
    tmpHat(tmpHat==0) = params.negClassValue;
    yHat = [yHat; num2cell(tmpHat',2)];
end

TF.yHat = yHat;


[muSleep_overallMetrics, muSleep_perRecMetrics] = eval_test_by(TF, "sample", params);
[muSleep_eventOverallMetrics, muSleep_eventPerRecMetrics] = eval_test_by(TF, "event", params); 

if apply_time_criteria

    critTF = TF;
    short_crit = cellfun(@(x) remove_invalid_labels(x, 3, inf, 1, -1, false), critTF.yTrue, "UniformOutput", false);
    long_crit = cellfun(@(x) remove_invalid_labels(x, 0, 15, 1, nan, true), short_crit, "UniformOutput", false);
    critTF.yTrue = long_crit;

    critTF.yhat = cellfun(@(x) remove_invalid_labels(x, 3, inf, 1, -1, false), critTF.yHat, "UniformOutput",false);
    [muSleep_overallMetrics, muSleep_perRecMetrics] = eval_test_by(critTF, "sample", params);

end

%% 6. Summarize the results


mdls = ["mUSleep"
        "uSleep";
          "LSTM";
          "RF";
          "SVM"];

% Overall      
overallSampleTable = struct2table([muSleep_overallMetrics,
                            uSleep_overallMetrics;
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
perRecSampleTable = perRecSampleTable([4,5,1:3],["model",perRecSampleTable.Properties.VariableNames(1:end-1)]);

disp(overallSampleTable)
perRecSampleTable.Variables

% Save
results = struct;
results.overall = overallSampleTable;
results.perRec = perRecSampleTable;
resultsFile = fullfile(params.outFolder,"sampleResults.mat");
save(resultsFile,"results");

write_results_table(overallSampleTable,perRecSampleTable,"sample",params)


%% 7. Summarize per event metrics
[RF_eventOverallMetrics, RF_eventPerRecMetrics] = eval_test_by(testRF, "event", params);
[SVM_eventOverallMetrics, SVM_eventPerRecMetrics] = eval_test_by(testSVM, "event", params);
[LSTM_eventOverallMetrics, LSTM_eventPerRecMetrics] = eval_test_by(testLSTM, "event", params);
[uSleep_eventOverallMetrics, uSleep_eventPerRecMetrics] = eval_test_by(testUsleep, "event", params);

% Overall      
overallEventTable = struct2table([muSleep_eventOverallMetrics;
                                  uSleep_eventOverallMetrics;
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
perRecEventTable = perRecEventTable([4,5,1:3],["model",perRecEventTable.Properties.VariableNames(1:end-1)]);

disp(overallEventTable)
perRecEventTable.Variables

% Save
results = struct;
results.overall = overallEventTable;
results.perRec = perRecEventTable;
resultsFile = fullfile(params.outFolder,"eventReults.mat");
save(resultsFile,"results");

write_results_table(overallEventTable,perRecEventTable,"event",params)


%% 8. Make informative plots of the test data
pause

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

    tmpStats = uSleep_perRecMetrics(uSleep_perRecMetrics.id==tmpId,1:end-1);
    info = cellfun(@(var,value) sprintf("%s=%.2f", var, value), ...
        tmpStats.Properties.VariableNames, table2cell(tmpStats));
    infoStr = strjoin(info,"        ");

    % Init figure
    figure(1); clf;
    yhdl = @(ax, x) ylabel(ax, x, "Rotation", 0, "HorizontalAlignment","right");

    subRows = 2+length(unique(test.model));
    subCols = 1;

    timePreds = linspace(0, 40, 40*60*params.labelsPerSec);

    % Plot the target labels
    tmpTargets = testUsleep.yTrue{i,:};
    ax1 = subplot(subRows, subCols, 1); hold on;
    plot_predictions(tmpTargets, timePreds, colors.BLUE, 1, ax1);

    xticklabels(ax1,[]);
    yhdl(ax1,"Target")

    % Plot Usleep probabilities
    probFile = sprintf("predictions//%i_hz//%s.mat",optHz,tmpId);
    probs = get_probs(probFile);
    ax2 = subplot(subRows, subCols, 2);
    timeProbs = linspace(0, 40, 40*60*optHz);
    plot_probs(probs, timeProbs, ax2);
    yline(ax2, optThres, "k-", "LineWidth",2);
    xticks(ax2,[]);
    xticklabels(ax2,[]);
    % legend(ax2,"off")
    yhdl(ax2, "P(Sleep)")

    for mIdx = 1:numel(mdls)
        % Get tmp model test results
        tmpModel = mdls(mIdx);
        tmpResults = tmp(tmp.model==tmpModel,:);

        % Plot Usleep predictions
        tmpPreds = cell2mat(tmpResults.yHat);
        ax2 = subplot(subRows,subCols,2+mIdx);
        plot_predictions(tmpPreds, timePreds, colors.BLUE, 1, ax2);
        if mIdx < numel(mdls)
            xticklabels(ax2,[]);
        else
            xlabel("Time [min]")
        end
        yhdl(ax2, tmpModel)
    end
    

    % Link all axes
    axs = findall(gcf,"Type","axes");
    linkaxes(axs)
    set(findall(gcf,"-property","FontSize"),"FontSize",12)
    set(gcf,"WindowState","fullscreen")

    sgtitle(sprintf("%s \n%s",tmpId,infoStr));

    %Save figure
    if saveOn
        figFile = fullfile(params.outFolder,"figures",sprintf("%s.png",tmpId));
        exportgraphics(gcf, figFile, "Resolution",300);
    end

end

%% 9. Make informative plots of the test data

% Transform usleep test to match
transUsleepTest = table;
transUsleepTest.id = testUsleep.id;
transUsleepTest.yTrue = cellfun(@(x) num2cell(x',1), testUsleep.yTrue);
transUsleepTest.yHat = cellfun(@(x) num2cell(x',1), testUsleep.yHat);

transmUsleepTest = table;
transmUsleepTest.id = TF.id;
transmUsleepTest.yTrue = cellfun(@(x) num2cell(x',1), TF.yTrue);
transmUsleepTest.yHat = cellfun(@(x) num2cell(x',1), TF.yHat);

% Join test results into common table
test = [transUsleepTest;
        transmUsleepTest];

test.model = repelem(["uSleep";"muSleep"],height(transUsleepTest),1);

uniqueIds = unique(test.id);

saveOn = 1;

for i = 1:numel(uniqueIds)

    tmpId = uniqueIds(i);
    tmp = test(test.id==tmpId,:);

    tmpStats = uSleep_perRecMetrics(uSleep_perRecMetrics.id==tmpId,1:end-1);
    info = cellfun(@(var,value) sprintf("%s=%.2f", var, value), ...
        tmpStats.Properties.VariableNames, table2cell(tmpStats));
    infoStr = strjoin(info,"        ");

    % Init figure
    figure(1); clf;
    yhdl = @(ax, x) ylabel(ax, x, "Rotation", 0, "HorizontalAlignment","right");

    subRows = 1+2*length(unique(test.model));
    subCols = 1;

    timePreds = linspace(0, 40, 40*60*params.labelsPerSec);

    % Plot the target labels
    tmpTargets = testUsleep.yTrue{i,:};
    ax1 = subplot(subRows, subCols, 1); hold on;
    plot_predictions(tmpTargets, timePreds, colors.BLUE, 1, ax1);

    xticklabels(ax1,[]);
    yhdl(ax1,"Target")

    % Plot Usleep probabilities
    probFile = sprintf("predictions//%i_hz//%s.mat",optHz,tmpId);
    probs = get_probs(probFile);
    ax2 = subplot(subRows, subCols, 2);
    timeProbs = linspace(0, 40, 40*60*optHz);
    plot_probs(probs, timeProbs, ax2);
    yline(ax2, optThres, "k-", "LineWidth",2);
    xticklabels(ax2,[]);
    % legend(ax2,"off")
    yticks([0,1])
    yhdl(ax2, "P(Sleep)")
    subtitle(ax2, "uSleep")
    

    tmpResults = tmp(tmp.model=="uSleep",:);


    % Plot Usleep predictions
    tmpPreds = cell2mat(tmpResults.yHat);
    ax3 = subplot(subRows,subCols, 3);
    plot_predictions(tmpPreds, timePreds, colors.BLUE, 1, ax3);
    xticklabels(ax3,[]);
    yhdl(ax3, "uSleep")
    

    % Plot mUsleep probabilities
    probs = TF{TF.id == tmpId,"probs"};
    probs = probs{1};
    ax4 = subplot(subRows, subCols, 4);
    timeProbs = linspace(0, 40, 40*60*1);
    if height(probs) < length(timeProbs)
    probs = [probs;
             nan(length(timeProbs)-height(probs),size(probs,2))];
    end
    area(ax4, timeProbs, probs(:,2), "EdgeColor","none");
    yline(ax4, tfOptThres, "k-", "LineWidth",2);
    xticklabels(ax4,[]);
    yticks([0,1])
    % legend(ax2,"off")
    subtitle(ax4, "muSleep")
    yhdl(ax4, "P(Sleep)")
    legend(ax4, "MS")
    
    
    tmpResults2 = tmp(tmp.model=="muSleep",:);
    
    % Plot Usleep predictions
    tmpPreds2 = cell2mat(tmpResults2.yHat);
    ax5 = subplot(subRows,subCols, 5);
    plot_predictions(tmpPreds2, timePreds, colors.BLUE, 1, ax5);
    yhdl(ax5, "muSleep")
    xlabel(ax5, "Time [min]")


    % Link all axes
    axs = findall(gcf,"Type","axes");
    linkaxes(axs)
    set(findall(gcf,"-property","FontSize"),"FontSize",12)
    set(gcf,"WindowState","fullscreen")

    sgtitle(sprintf("%s \n%s",tmpId,infoStr));

    %Save figure
    if saveOn
        figFile = fullfile(params.outFolder,"figures",sprintf("%s.png",tmpId));
        exportgraphics(gcf, figFile, "Resolution",300);
    end
    
end


