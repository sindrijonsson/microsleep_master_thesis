% Main script

setup

% 0. Set the parameters of the main script
params = struct;
params.outFolder = "new_main_comparison";
params.filePaths = strcat("data/", ls("data/*.mat"));
params.fileIds = params.filePaths.extractAfter("data/").extractBefore(".mat");
params.seed = load("seed.mat");
params.split = parse_json("..\splits\skorucack_splits.json");
params.masterFile = fullfile(params.outFolder, "master.mat");

% Label specifics
params.posClassValue = 1;
params.negClassValue = -1;
params.trainTargets = "wakeED_vs_restMSE";
params.testTargets = "wakeED_vs_restMSE";

% Time domain values
params.secPerLabel   = 1;
params.labelsPerSec  = 1 / params.secPerLabel;
params.windowSizeSec = 1;
params.windowStepSec = 1;
params.downsampleFcn = @(x) my_median(x); % down sampling technique
params.minDurationSec = 0;
params.maxDurationSec = inf;
params.skipSingles = true;
params.replaceInvalidPredictions = nan;

% Features
params.zeroPad = zeros(1, 0); % 40 samples account for the first and last two samples lost with the moving window
params.numFeatures = 14;
params.pburgWindowSize = 1;
params.pburgFreqRes = 0.2;


% Model specifics
params.retrain = [""];%["RF","SVM","LSTM"];
params.windowSizeLSTM = 9;

% U-sleep specifics
usleep = struct;
usleep.splitValue = 0.5;
usleep.hz = [32, 16, 8, 1];
usleep.thresholds = (0.025:0.025:1)';
usleep.outfile = fullfile(params.outFolder, "uSleep.mat");
usleep.evalMetric = "f1";
params.usleep = usleep;


%% 0. Generate the folder of the analysis if it doesn't exist
% Store the params in the folder and all of the the models
init_project(params);

%% 1. Generate master table with labels and feature from Skorucak 2020
master = init_master(params);

% Save labels for model training
% [idx, id] = findgroups(master.id);
% recLabels = splitapply(@(x) {x}, master.(params.trainTargets), idx);
% cellfun(@(x, id) save(sprintf("..\\edf_data\\%s_new.mat",id), "x"), recLabels, id);

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

cvStats = summarize_cv(cvUsleep,params);

[optUsleep, testUsleep, optThres] = train_and_test_usleep(uSleepPredictions, optHz, optMethod, params);

[uSleep_overallMetrics, uSleep_perRecMetrics] = eval_test_by(testUsleep, "sample", params);


%% 6, Transfer learning 

% Let's evaluate transfer learning results
testMuSleep = load("transfer_learning_new.mat");
tlOptThres = testMuSleep.optThres;
testMuSleep = rmfield(testMuSleep,"optThres");
testMuSleep.id = string(testMuSleep.id);
% TF.yHat = TF.yHat';
testMuSleep.yTrue = testMuSleep.yTrue';
testMuSleep.probs = testMuSleep.probs';
testMuSleep = struct2table(testMuSleep);

% Sort according to testUsleep
rows=arrayfun(@(x) find(x==testMuSleep.id), testUsleep.id);
testMuSleep = testMuSleep(rows,:);
testMuSleep.yTrue = testUsleep.yTrue;

yHat = [];
for i = 1:height(testMuSleep)
    tmpHat = double(testMuSleep.probs{i}(:,2) >= tlOptThres);
    tmpHat(tmpHat==0) = params.negClassValue;
    yHat = [yHat; num2cell(tmpHat',2)];
end

testMuSleep.yHat = yHat;

[mUSleep_overallMetrics, mUSleep_perRecMetrics] = eval_test_by(testMuSleep, "sample", params);

%% 6.2 Malafeev
% Let's evaluate transfer learning results
testMalafeev = load("malafeev42_new.mat");

testMalafeev.id = string(testMalafeev.id);
testMalafeev.yHat = testMalafeev.yHat';
testMalafeev.yTrue = testMalafeev.yTrue';
testMalafeev.probs = testMalafeev.probs';
testMalafeev = struct2table(testMalafeev);

% Sort according to testUsleep
rows=arrayfun(@(x) find(x==testMalafeev.id), testUsleep.id);
testMalafeev = testMalafeev(rows,:);
testMalafeev.yTrue = testUsleep.yTrue;

yHat = [];
for i = 1:height(testMalafeev)
    tmpHat = testMalafeev.yHat{i};
    tmpHat = double(tmpHat);
    tmpHat(tmpHat==0) = params.negClassValue;
    yHat = [yHat; num2cell(tmpHat,2)];
end

testMalafeev.yHat = yHat;

[malafeev_overallMetrics, malafeev_perRecMetrics] = eval_test_by(testMalafeev, "sample", params);

%% 6.3 Self-supervised learning 

% Let's evaluate SSL results
testSSL = load("main_comparison/ssl.mat");
sslOptThres = testSSL.optThres;
testSSL = rmfield(testSSL,"optThres");
testSSL.id = string(testSSL.id);
% TF.yHat = TF.yHat';
testSSL.yTrue = testSSL.yTrue';
testSSL.probs = testSSL.probs';
testSSL = struct2table(testSSL);

% Sort according to testUsleep
rows=arrayfun(@(x) find(x==testSSL.id), testUsleep.id);
testSSL = testSSL(rows,:);
testSSL.yTrue = testUsleep.yTrue;

yHat = [];
for i = 1:height(testSSL)
    tmpHat = double(testSSL.probs{i} > sslOptThres);
    tmpHat(tmpHat==0) = params.negClassValue;
    yHat = [yHat; num2cell(tmpHat,2)];
end

testSSL.yHat = yHat;

[ssl_overallMetrics, ssl_perRecMetrics] = eval_test_by(testSSL, "sample", params);

%% 6. Summarize the results

% Overall      
overallSampleTable = struct2table([mUSleep_overallMetrics;
                            uSleep_overallMetrics;
                            ssl_overallMetrics;
                            LSTM_overallMetrics;
                            RF_overallMetrics;
                            SVM_overallMetrics; ...
                            malafeev_overallMetrics]);
                        
overallSampleTable.model = mdls;
overallSampleTable = overallSampleTable(:,["model",overallSampleTable.Properties.VariableNames(1:end-1)]);


% Per rec
perRecSample = {mUSleep_perRecMetrics,
                uSleep_perRecMetrics,
                ssl_perRecMetrics,
                LSTM_perRecMetrics,
                RF_perRecMetrics,
                SVM_perRecMetrics,
                malafeev_perRecMetrics};
perRecSampleTable = table;
for i = 1:numel(perRecSample)
    var = perRecSample{i};
    [m, s] = summarize_perRecMetrics(var, "sample");
    uSleepInfo = cellfun(@(m,s) sprintf("%.2f +/- %.2f",m,s), struct2cell(m),struct2cell(s));
    entry = cell2struct(cellstr(uSleepInfo),fieldnames(m));
    perRecSampleTable = [perRecSampleTable; struct2table(entry)];
end
perRecSampleTable.model = mdls;
perRecSampleTable = perRecSampleTable(:,["model",perRecSampleTable.Properties.VariableNames(1:end-1)]);

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
[ssl_eventOverallMetrics, ssl_eventPerRecMetrics] = eval_test_by(testSSL, "event", params); 
[LSTM_eventOverallMetrics, LSTM_eventPerRecMetrics] = eval_test_by(testLSTM, "event", params);
[uSleep_eventOverallMetrics, uSleep_eventPerRecMetrics] = eval_test_by(testUsleep, "event", params);
[muSleep_eventOverallMetrics, muSleep_eventPerRecMetrics] = eval_test_by(testMuSleep, "event", params); 
[malafeev_eventOverallMetrics, malafeev_eventPerRecMetrics] = eval_test_by(testMalafeev, "event", params); 

% Overall      
overallEventTable = struct2table([muSleep_eventOverallMetrics;
                                  uSleep_eventOverallMetrics;
                                  ssl_eventOverallMetrics;
                                  LSTM_eventOverallMetrics;
                                  RF_eventOverallMetrics;
                                  SVM_eventOverallMetrics; ...
                                  malafeev_eventOverallMetrics]);
                        
overallEventTable.model = mdls;
overallEventTable = overallEventTable(:,["model",overallEventTable.Properties.VariableNames(1:end-1)]);

% Per rec
perRecEvent = {mUSleep_perRecMetrics,
                uSleep_perRecMetrics,
                ssl_perRecMetrics,
                LSTM_perRecMetrics,
                RF_perRecMetrics,
                SVM_perRecMetrics,
                malafeev_perRecMetrics};
perRecEventTable = table;
for i = 1:numel(perRecEvent)
    var = perRecEvent{i};
    [m, s] = summarize_perRecMetrics(var, "event");
    uSleepInfo = cellfun(@(m,s) sprintf("%.2f +/- %.2f",m,s), struct2cell(m),struct2cell(s));
    entry = cell2struct(cellstr(uSleepInfo),fieldnames(m));
    perRecEventTable = [perRecEventTable; struct2table(entry)];
end
perRecEventTable.model = mdls
perRecEventTable = perRecEventTable(:,["model",perRecEventTable.Properties.VariableNames(1:end-1)]);

disp(overallEventTable)
perRecEventTable.Variables

% Save
results = struct;
results.overall = overallEventTable;
results.perRec = perRecEventTable;
resultsFile = fullfile(params.outFolder,"eventReults.mat");
save(resultsFile,"results");

write_results_table(overallEventTable,perRecEventTable,"event",params)


%% Stop before plotting
return

%% 8. Make informative plots of the test data for U-Sleep
% Targets
% Probabilietes
% Predictions

% Transform usleep test to match
transUsleepTest = table;
transUsleepTest.id = testUsleep.id;
transUsleepTest.yTrue = cellfun(@(x) num2cell(x',1), testUsleep.yTrue);
transUsleepTest.yHat = cellfun(@(x) num2cell(x',1), testUsleep.yHat);

transmUsleepTest = table;
transmUsleepTest.id = testMuSleep.id;
transmUsleepTest.yTrue = cellfun(@(x) num2cell(x',1), testMuSleep.yTrue);
transmUsleepTest.yHat = cellfun(@(x) num2cell(x',1), testMuSleep.yHat);

% Join test results into common table
test = [transUsleepTest;
        transmUsleepTest];

test.model = repelem(["uSleep";"muSleep"],height(transUsleepTest),1);

uniqueIds = unique(test.id);

saveOn = 0;

msColor = colors.GREEN;

for i = 1:numel(uniqueIds)
%     i = randi(numel(uniqueIds),1,1);

    tmpId = uniqueIds(i);
    tmp = test(test.id==tmpId,:);

    uSleepStats = uSleep_perRecMetrics(uSleep_perRecMetrics.id==tmpId,1:end-1);
    uSleepInfo = cellfun(@(var,value) sprintf("%s=%.2f", var, value), ...
        uSleepStats.Properties.VariableNames, table2cell(uSleepStats));
    uSleepInfo = uSleepInfo(contains(uSleepInfo,["recall","precision","f1"]));

    % Init figure
    figure(1); clf;
    yhdl = @(ax, x) ylabel(ax, x, "Rotation", 0, "HorizontalAlignment","right");
    yyhdl = @(ax, x) ylabel(ax, x, ...
                            "Rotation", 0, ...
                            "HorizontalAlignment","left", ...
                            "VerticalAlignment","middle",...
                            "Color","k"); 
    

    subRows = 1+2*length(unique(test.model));
    subCols = 1;

    timePreds = linspace(0, 40, 40*60*params.labelsPerSec);

    % Plot the target labels
    tmpTargets = testUsleep.yTrue{i,:};
    ax1 = subplot(subRows, subCols, 1); hold on;
    plot_predictions(tmpTargets, timePreds, msColor, 1, ax1);

    xticklabels(ax1,[]);
    yhdl(ax1,"Target")
    title(ax1, tmpId+".mat")
    box(ax1,"on")

    % Plot Usleep probabilities
    probFile = sprintf("predictions//%i_hz//%s.mat",optHz,tmpId);
    probs = get_probs(probFile);
    ax2 = subplot(subRows, subCols, 2); 
    timeProbs = linspace(0, 40, 40*60*optHz);
    plot_probs(probs, timeProbs, ax2);
    if contains(lower(optMethod),"argmax")
        yline(ax2, 0.5, "k-", "LineWidth",1);
    else
        yline(ax2, optThres, "k-", "LineWidth",1);
        
    end
    xticklabels(ax2,[]);
    % legend(ax2,"off")
%     yticks([0,1])
    yhdl(ax2, "P(Sleep)")
    subtitle(ax2, "uSleep", "FontWeight","bold")
    

    tmpResults = tmp(tmp.model=="uSleep",:);


    % Plot Usleep predictions
    tmpPreds = cell2mat(tmpResults.yHat);
    ax3 = subplot(subRows,subCols, 3);
    plot_predictions(tmpPreds, timePreds, msColor, 1, ax3);
    xticklabels(ax3,[]);
    yhdl(ax3, "Predictions")
    ax3.YTick=[]; ax3.YTickLabels=[];
    yyaxis right
    yyhdl(ax3, uSleepInfo)
    ax3.YTick=[];
    ax3.YTickLabels=[];
      
    % ================================================================= %
    %                               mUSleep                             %
    % ================================================================= %

    muSleepStats = mUSleep_perRecMetrics(uSleep_perRecMetrics.id==tmpId,1:end-1);
    muSleepInfo = cellfun(@(var,value) sprintf("%s=%.2f", var, value), ...
        uSleepStats.Properties.VariableNames, table2cell(muSleepStats));
    muSleepInfo = muSleepInfo(contains(muSleepInfo,["recall","precision","f1"]));


    % Plot mUsleep probabilities
    probs = testMuSleep{testMuSleep.id == tmpId,"probs"};
    probs = probs{1};
    ax4 = subplot(subRows, subCols, 4);
    timeProbs = linspace(0, 40, 40*60*1);
    if height(probs) < length(timeProbs)
    probs = [probs;
             nan(length(timeProbs)-height(probs),size(probs,2))];
    end
    area(ax4, timeProbs, probs(:,2), "EdgeColor","none");
    yline(ax4, tlOptThres, "k-", "LineWidth", 1);
    xticklabels(ax4,[]);
%     yticks([0,1])
    % legend(ax2,"off")
    subtitle(ax4, "muSleep", "FontWeight", "bold")
    yhdl(ax4, "P(Sleep)")
    legend(ax4, "MS", "Location","northwest");
    
    
    tmpResults2 = tmp(tmp.model=="muSleep",:);
    
    % Plot Usleep predictions
    tmpPreds2 = cell2mat(tmpResults2.yHat);
    ax5 = subplot(subRows,subCols, 5);
    plot_predictions(tmpPreds2, timePreds, msColor, 1, ax5);
    xlabel(ax5, "Time [min]");
    yhdl(ax5, "Predictions")
    ax3.YTick=[]; ax3.YTickLabels=[];
    yyaxis right
    yyhdl(ax5,muSleepInfo)
    box(ax5,"on")
    ax5.YTick=[];
    ax5.YTickLabels=[];

    % Link all axes
    axs = findall(gcf,"Type","axes");
    linkaxes(axs)
    set(findall(gcf,"-property","FontSize"),"FontSize",12)
    set(gcf,"WindowState","fullscreen")

    % Set all yaxis colors to black
    for n = 1:numel(axs)
        ax = axs(n);
        for yi = 1:numel(ax.YAxis)
            ax.YAxis(yi).Color="k";
        end
    end

    %Save figure
    if saveOn
        figFile = fullfile(params.outFolder,"figures",sprintf("%s.png", tmpId));
        exportgraphics(gcf, figFile, "Resolution",300);
    end
   pause 
end

%% 8. Make informative plots of the test data for U-Sleep
% Transform usleep test to match
transMalafeevTest = table;
transMalafeevTest.id = testMalafeev.id;
transMalafeevTest.yTrue = cellfun(@(x) num2cell(x',1), testMalafeev.yTrue);
transMalafeevTest.yHat = cellfun(@(x) num2cell(x',1), testMalafeev.yHat);

% Join test results into common table
testAll = [transmUsleepTest;
    transUsleepTest;
    testLSTM;
    testRF;
    testSVM;
    transMalafeevTest];

testAll.model = repelem(mdls,height(transUsleepTest),1);

uniqueIds = unique(testAll.id);

saveOn = 0;

for i = 1:numel(uniqueIds)
    i=randi(numel(uniqueIds),1,1);

    tmpId = uniqueIds(i);
    tmp = testAll(testAll.id==tmpId,:);
    
    
    % Init figure
    figure(1); clf;
    yhdl = @(ax, x) ylabel(ax, x, ...
                           "Rotation", 0, ...
                           "HorizontalAlignment","right", ...
                           "Interpreter","none");
    
    yt = @(ax) set(ax, "YTick", [], "YTickLabels",[]);

    subRows = 1+length(unique(testAll.model));
    subCols = 1;

    timePreds = linspace(0, 40, 40*60*params.labelsPerSec);

    % Plot the target labels
    tmpTargets = tmp.yTrue{1,:};
    ax1 = subplot(subRows, subCols, 1); hold on;
    plot_predictions(tmpTargets, timePreds, colors.BLUE, 1, ax1);
    lgdHdl =[]; lgdName = [];
    p=findall(ax1.Children,"Type","Patch");
    % TO INCLUDE NAN TARGETS UNCOMMENT BELOW:
%     if ~isempty(p); lgdHdl = [p(1)]; lgdName = ["MS"]; end
    plot_predictions(isnan(tmpTargets), timePreds, [220,220,220]/256, 0.8, ax1);
%     p2=findall(ax1.Children,"Type","Patch");
%     if length(p2) > length(p); lgdHdl = [lgdHdl; p2(1)]; lgdName = [lgdName; "InvalidMS"]; end
%     if ~isempty(lgdHdl); legend(lgdHdl, lgdName, "Location","northwest"); end
    xticklabels(ax1,[]);
    yhdl(ax1,"Target")
    yt(ax1)
    subtitle(tmpId, "FontWeight","bold")

    
    for mIdx = 1:numel(mdls)
        % Get tmp model test results
        tmpModel = mdls(mIdx);
        tmpResults = tmp(tmp.model==tmpModel,:);

        % Plot Usleep predictions
        tmpPreds = cell2mat(tmpResults.yHat);
        axMdl = subplot(subRows,subCols,1+mIdx);
        plot_predictions(tmpPreds, timePreds, msColor, 1, axMdl);
        if mIdx < numel(mdls)
            xticklabels(axMdl,[]);
        else
            xlabel("Time [min]")
        end
        yhdl(axMdl, tmpModel)
        yt(axMdl);

        % Stats
        [~,tmpStats] = eval_test_by(tmpResults,"sample",params);
        info = cellfun(@(var,value) sprintf("%s=%.2f", var, value), ...
            tmpStats.Properties.VariableNames, table2cell(tmpStats));
        info = info(contains(info,["recall","precision","f1"]));


        yyaxis right
        yyhdl(axMdl, info)
        box(axMdl,"on")
        axMdl.YTick=[];
        axMdl.YTickLabels=[];

    end
    
    % Clean up

    % Link all axes
    axs = findall(gcf,"Type","axes");
    linkaxes(axs)
    set(findall(gcf,"-property","FontSize"),"FontSize",12)
    set(gcf,"WindowState","fullscreen")
    
    for j = 1:numel(axs)
        box(axs(j), "on")
    end


    % Set all yaxis colors to black
    for n = 1:numel(axs)
        ax = axs(n);
        for yi = 1:numel(ax.YAxis)
            ax.YAxis(yi).Color="k";
        end
    end


    %Save figure
    if saveOn
        figFile = fullfile(params.outFolder,"figures",sprintf("%s.png",tmpId));
        exportgraphics(gcf, figFile, "Resolution",300);
    end
    pause

end
