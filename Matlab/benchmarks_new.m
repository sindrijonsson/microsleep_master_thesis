close all; clear all; clc
% Main script

setup

% 0. Set the parameters of the main script
params = struct;
params.outFolder = "benchmark_test_new";
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
params.secPerLabel   = 0.2;
params.labelsPerSec  = 1 / params.secPerLabel;
params.windowSizeSec = 0.2;
params.windowStepSec = 0.2;
params.downsampleFcn = @(x) my_median(x); % down sampling technique
params.minDurationSec = 1;
params.maxDurationSec = inf;
params.skipSingles = false;
params.replaceInvalidPredictions = nan;

% Features
params.zeroPad = zeros(1, 80); % 40 samples account for the first and last two samples lost with the moving window
params.numFeatures = 14;
params.pburgWindowSize = 1;
params.pburgFreqRes = 0.2;

% Model specifics
params.retrain = ["RF","SVM","LSTM"];
params.windowSizeLSTM = 45;

%% 0. Generate the folder of the analysis if it doesn't exist
% Store the params in the folder and all of the the models
init_project(params);

%% 1. Generate master table with labels and feature from Skorucak 2020
master = init_master(params);

%% Iterate through the different grouping of classes
groupings = master.Properties.VariableNames(contains(master.Properties.VariableNames,"_vs_"));

for i = 1:numel(groupings)
    tmpGroups = groupings{i}
    params.trainTargets = "biWake_vs_biMSE";
    params.testTargets = tmpGroups;

    if i>1; params.retrain = [""]; end

    %% 2. Train and evaluate RF model

    [RF, testRF] = train_and_test_model("RF", params, master);
    [RF_overallMetrics, RF_perRecMetrics] = eval_test_benchmark_by(testRF, "sample", params);

    %% 3. Train and evaluate SVM model

    [SVM, testSVM] = train_and_test_model("SVM", params, master);
    [SVM_overallMetrics, SVM_perRecMetrics] = eval_test_benchmark_by(testSVM, "sample", params);

    %% 4. Train and eval LSTM model

    [LSTM, testLSTM] = train_and_test_lstm(params, master);
    [LSTM_overallMetrics, LSTM_perRecMetrics] = eval_test_benchmark_by(testLSTM, "sample", params);

    %% 5. Summarize the results
    mdls = ["LSTM";
        "RF";
        "SVM"];

    % Overall
    overallSampleTable = struct2table([LSTM_overallMetrics;
        RF_overallMetrics;
        SVM_overallMetrics]);

    overallSampleTable.model = mdls;
    overallSampleTable = overallSampleTable(:,["model",overallSampleTable.Properties.VariableNames(1:end-1)]);


    % Per rec
    perRecSample = who('*_perRecMetrics');
    perRecSampleTable = table;
    for i = 1:numel(perRecSample)
        var = perRecSample{i};
        [m, s] = summarize_benchmark_perRecMetrics(eval(var), "sample");
        info = cellfun(@(m,s) sprintf("%.2f +/- %.2f",m,s), struct2cell(m),struct2cell(s));
        entry = cell2struct(cellstr(info),fieldnames(m));
        entry.model = string(var).extractBefore("_");
        perRecSampleTable = [perRecSampleTable; struct2table(entry)];
    end
    perRecSampleTable = perRecSampleTable(:,["model",perRecSampleTable.Properties.VariableNames(1:end-1)]);
    perRecSampleTable.model = mdls;

    disp(overallSampleTable)
    perRecSampleTable.Variables

    % Save
    results = struct;
    results.overall = overallSampleTable;
    results.perRec = perRecSampleTable;
    resultsFile = fullfile(params.outFolder,"sampleResults.mat");
    save(resultsFile,"results");

    write_results_table(overallSampleTable,perRecSampleTable,"sample",params,params.testTargets)
end