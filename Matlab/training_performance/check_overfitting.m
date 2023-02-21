% Check if training data is different from test data in performcne
% How do the models generalize?
setup

%% Be sure to run main first!
% main

%% 
masterCopy = master;
masterCopy.train = ~master.train;

%% U-Sleep
trainUSleep = uSleepPredictions((uSleepPredictions.hz==optHz) & ...
                                 uSleepPredictions.train==1,["testTargets", optMethod, "id", "probs"]);
trainUSleep = renamevars(trainUSleep, ["testTargets", optMethod], ["yTrue","yHat"]);
perRecProbs = trainUSleep{:, "probs"};

if contains(optMethod,"max")
    perRecProbs = cellfun(@(x) max(x(2:end,:),[],1), perRecProbs, 'UniformOutput',false);
elseif contains(optMethod, "sum")
    perRecProbs = cellfun(@(x) sum(x(2:end,:),[],1), perRecProbs, 'UniformOutput',false);
else
    perRecProbs = perRecProbs;
end
trainUSleep.probs = perRecProbs;

[trainUsleep_overall, trainUsleep_perRec] = eval_test_by(trainUSleep, "sample", params);

%% RF
[~, trainRF] = train_and_test_model("RF",params,masterCopy);
[trainRF_overall, trainRF_perRec] = eval_test_by(trainRF, "sample", params);

%% SVM
[~, trainSVM] = train_and_test_model("SVM",params,masterCopy);
[trainSVM_overall, trainSVM_perRec] = eval_test_by(trainSVM, "sample", params);

%% LSTM
[~, trainLSTM] = train_and_test_lstm(params,masterCopy);
[trainLSTM_overall, trainLSTM_perRec] = eval_test_by(trainLSTM, "sample", params);

%% mU-Sleep (transfer learning)
% Let's evaluate transfer learning results
trainMuSleep = load("training_performance\\transfer_learning_new.mat");
tlOptThres = trainMuSleep.optThres;
trainMuSleep = rmfield(trainMuSleep,"optThres");
trainMuSleep.id = string(trainMuSleep.id);
% TF.yHat = TF.yHat';
trainMuSleep.yTrue = trainMuSleep.yTrue';
trainMuSleep.probs = trainMuSleep.probs';
trainMuSleep = struct2table(trainMuSleep);

% Sort according to testUsleep
rows=arrayfun(@(x) find(x==trainMuSleep.id), trainUSleep.id);
trainMuSleep = trainMuSleep(rows,:);
trainMuSleep.yTrue = trainUSleep.yTrue;

% tfOptThres = 0.275;
% tfOptThres = 0.45848149061203003;
% tlOptThres = 0.275;

yHat = [];
for i = 1:height(trainMuSleep)
    tmpHat = double(trainMuSleep.probs{i}(:,2) > tlOptThres);
    tmpHat(tmpHat==0) = params.negClassValue;
    yHat = [yHat; num2cell(tmpHat',2)];
end

trainMuSleep.yHat = yHat;

[trainMuSleep_overall, trainMuSleep_perRec] = eval_test_by(trainMuSleep, "sample", params);

%% Malafeev CNN_16s
trainMalafeev = load("training_performance\malafeev42_new.mat");

% trainMalafeev = rmfield(trainMalafeev,"probs");
trainMalafeev.id = string(trainMalafeev.id);
trainMalafeev.yHat = trainMalafeev.yHat';
trainMalafeev.yTrue = trainMalafeev.yTrue';
trainMalafeev.probs = trainMalafeev.probs';
trainMalafeev = struct2table(trainMalafeev);

% Sort according to testUsleep
rows=arrayfun(@(x) find(x==trainMalafeev.id), trainUSleep.id);
trainMalafeev = trainMalafeev(rows,:);
trainMalafeev.yTrue = trainUSleep.yTrue;

yHat = [];
for i = 1:height(trainMalafeev)
    tmpHat = trainMalafeev.yHat{i};
    tmpHat = double(tmpHat);
    tmpHat(tmpHat==0) = params.negClassValue;
    yHat = [yHat; num2cell(tmpHat,2)];
end

trainMalafeev.yHat = yHat;

[trainMalafeev_overall, trainMalafeev_perRec] = eval_test_by(trainMalafeev, "sample", params);

%% SSL training

% Let's evaluate SSL results
trainSSL = load("training_performance\ssl_new.mat");
sslOptThres = trainSSL.optThres;
trainSSL = rmfield(trainSSL,"optThres");
trainSSL.id = string(trainSSL.id);
% TF.yHat = TF.yHat';
trainSSL.yTrue = trainSSL.yTrue';
trainSSL.probs = trainSSL.probs';
trainSSL = struct2table(trainSSL);

% Sort according to testUsleep
rows=arrayfun(@(x) find(x==trainSSL.id), trainUSleep.id);
trainSSL = trainSSL(rows,:);
trainSSL.yTrue = trainUSleep.yTrue;

yHat = [];
for i = 1:height(trainSSL)
    tmpHat = double(trainSSL.probs{i} > sslOptThres);
    tmpHat(tmpHat==0) = params.negClassValue;
    yHat = [yHat; num2cell(tmpHat,2)];
end

trainSSL.yHat = yHat;

[trainSSL_overall, trainSSL_perRec] = eval_test_by(trainSSL, "sample", params);

%% Make table between train and test

% Overall      
testOverallTable = struct2table([mUSleep_overallMetrics;
                            uSleep_overallMetrics;
                            ssl_overallMetrics
                            LSTM_overallMetrics;
                            RF_overallMetrics;
                            SVM_overallMetrics; 
                            malafeev_overallMetrics]);
                        
testOverallTable.model = mdls;
testOverallTable = testOverallTable(:,["model",testOverallTable.Properties.VariableNames(1:end-1)]);
testOverallTable.dataset = repelem("test", height(testOverallTable), 1)

trainOverallTable = struct2table([trainMuSleep_overall;
                            trainUsleep_overall; ...
                            trainSSL_overall; ...
                            trainLSTM_overall;
                            trainRF_overall;
                            trainSVM_overall; ...
                            trainMalafeev_overall]);
trainOverallTable.model = mdls;
trainOverallTable = trainOverallTable(:,["model",trainOverallTable.Properties.VariableNames(1:end-1)]);
trainOverallTable.dataset = repelem("train", height(trainOverallTable), 1);

overall = [testOverallTable; trainOverallTable];

% Per rec
testPerRec = {mUSleep_perRecMetrics,
                uSleep_perRecMetrics,
                ssl_perRecMetrics,
                LSTM_perRecMetrics,
                RF_perRecMetrics,
                SVM_perRecMetrics,
                malafeev_perRecMetrics};
testPerRecTable = table;
for i = 1:numel(testPerRec)
    var = testPerRec{i};
    [m, s] = summarize_perRecMetrics(var, "sample");
    sEntry = struct2table(s);
    vars = sEntry.Properties.VariableNames;
    sEntry.Properties.VariableNames = strcat("sem_",vars);
    testPerRecTable = [testPerRecTable; struct2table(m), sEntry];
end
testPerRecTable.dataset = repelem("test",height(testPerRecTable),1);
testPerRecTable.model = mdls;

trainPerRec =  {trainMuSleep_perRec,
                trainUsleep_perRec,
                trainSSL_perRec,
                trainLSTM_perRec,
                trainRF_perRec,
                trainSVM_perRec,
                trainMalafeev_perRec};

trainPerRecTable = table;
for i = 1:numel(trainPerRec)
    var = trainPerRec{i};
    [m, s] = summarize_perRecMetrics(var, "sample");
    sEntry = struct2table(s);
    vars = sEntry.Properties.VariableNames;
    sEntry.Properties.VariableNames = strcat("sem_",vars);
    trainPerRecTable = [trainPerRecTable; struct2table(m), sEntry];
end
trainPerRecTable.dataset = repelem("train",height(trainPerRecTable),1);
trainPerRecTable.model = mdls;

perRec = [testPerRecTable; trainPerRecTable];


%% Plot results as overall and per rec mean +/- sem

y = "f1";

figure(100); clf

for i = 1:numel(mdls)
    
    
    ax=subplot(2,4,i); hold on;
    
    % Overall
    tmpOverall = overall(overall.model==mdls(i),:);
    tmpOverall.dataset = categorical(tmpOverall.dataset);
    o=plot(tmpOverall.dataset, tmpOverall.(y), ...
        "Color", colors.BLUE,"LineStyle","-","Marker","*");
    
    % Per rec
    tmpPerRec = perRec(perRec.model==mdls(i),:);
    tmpPerRec.dataset = categorical(tmpPerRec.dataset);
    p=errorbar(tmpPerRec.dataset, tmpPerRec.(y), tmpPerRec.("sem_"+y), ...
        "Color", colors.ORANGE,"LineStyle","-", ...
        "Marker","o", "MarkerFaceColor",colors.ORANGE);
    
    ylim(ax,[0,1])
    grid(ax,"on")
    title(ax,mdls(i),"Interpreter","none")
    box(ax,"on")
end
