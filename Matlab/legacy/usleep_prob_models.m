files = string(ls(".\probs\"));
files = files(files.endsWith(".mat"));
files = strcat(".\probs\",files)

%%

[idx, rec] = findgroups(master.id);
recLabels = splitapply(@(x) {x}, master{:,"wakeED_vs_restMSE"}, idx);

%%
pTable = table;
for i = 1:numel(files)

    id = files(i).extractBefore(".mat").extractAfter(".\probs\");
    
    probData = load(files(i));
    probs = probData.probs;
    rp = reshape(probs,[],size(probs,2),1);

    y = recLabels{rec == id};
    
    tmp = table;
    tmp.id = repelem(id,length(y),1);
    tmp.x = num2cell(rp',2);
    tmp.y = y;
    tmp.train = master.train(master.id == id);

    pTable = [pTable; tmp];    
end

%%

trainData = pTable(pTable.train,:);

yTrain = trainData{:,"y"};

% Collect negative and positive classes
posClasses = find(yTrain==params.posClassValue);
negClasses = find(yTrain==params.negClassValue);

% Select random negative classes (balance with positive classes)
rng(params.seed.generator);
negIdx = randi(length(negClasses),[length(posClasses),1]);
useNegClasses = negClasses(negIdx);

useIdx = [posClasses; useNegClasses];
trainX = vertcat(trainData.x{useIdx});
trainY = categorical(yTrain(useIdx));

% -------------------------------------------------------------------------
% Train the specified model (either RF or SVM) using the training data



model = TreeBagger(1, trainX, trainY, ...
        "Method", "classification", ...
        "SampleWithReplacement", "on", ...
        "InBagFraction", 1, ...
        "OOBPrediction","on")%, ...
%         "NumPredictorsToSample", 4, ...
%         "MinLeafSize", 1);
    

%%
testData = pTable(~pTable.train,:);
testTable = table;
testTable.id = testData.id;
testTable.X = vertcat(testData.x{:});
testTable.Y = vertcat(testData{:,"y"});
    
% Make predictions per recording with post-processing
[idx, id] = findgroups(testTable.id);

perRecPredictions = splitapply(@(x) {my_predict(x,model,params)}, ...
                              testTable.X, idx);

perRecTargets = splitapply(@(x) num2cell(x,1), testTable.Y, idx);

testResults = table;
testResults.id = id;
testResults.yTrue = perRecTargets;
testResults.yHat  = perRecPredictions;

function yHatRaw = my_predict(X, model, params)

yHatCategorical = categorical(predict(model, X));
yHatRaw = grp2idx(yHatCategorical)-1;
yHatRaw(yHatRaw==0) = params.negClassValue;

end
