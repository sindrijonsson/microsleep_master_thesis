%%
master = load("master_zeropad.mat");
ref = load("Skorucak2020\Supplementary_Material.mat",'-mat');

%%
data = master.checkPoint.dataTable;
trainData = data(data.train,:);
testData = data(~data.train,:);

%% Use only bilateral
tmp = trainData(:,["patient","sample","train","features"]);
tmp.trainTarget = trainData.biWake_vs_biMSE;

% Collect negative and positive classes
posClasses = find(tmp.trainTarget==1);
negClasses = find(tmp.trainTarget==-1);

% Select random negative classes (balance with positive classes)
seed = load("seed.mat");
rng(seed.generator);
negIdx = randi(length(negClasses),[length(posClasses),1]);
useNegClasses = negClasses(negIdx);

useIdx = [posClasses; useNegClasses];
trainX = vertcat(tmp.features{useIdx});
trainY = categorical(tmp.trainTarget(useIdx));

%% Train model

model = train_model(trainX, trainY, "RF");
% model = svmModel;

%% Test model

testTable.rec = testData.patient;
testTable.X = vertcat(testData.features{:});
testTable.Y = vertcat(testData.biWake_vs_biMSE);

%% Predict on each recording
[idx, rec] = findgroups(testTable.rec);

perRecPredictions = splitapply(@(x) predict_and_post_process(x,model,1), ...
                              testTable.X, idx);

perRecTargets = splitapply(@(x) num2cell(x,1), testTable.Y, idx);

perRecPerformance = cellfun(@(x,y) calc_performance_metrics(x,y), ...
                            perRecTargets, perRecPredictions);

perRecOrgLabels = arrayfun(@(x)get_labels(sprintf("data/%s.mat",x),0),rec);

overallPerformance = calc_performance_metrics( ...
                                              vertcat(perRecTargets{:}), ...
                                              vertcat(perRecPredictions{:}) ...
                                              );

%% Summarize results
