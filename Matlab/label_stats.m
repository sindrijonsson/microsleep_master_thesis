%% 

load("main_comparison\params.mat","params")

rawTable = table;
postTable = table;
grpTable = table;

flipStruct = @(s) structfun(@(x) x', s, "UniformOutput",false);

for fIdx = 1:numel(params.filePaths)

    tmpFile = params.filePaths(fIdx);
    tmpId = params.fileIds(fIdx);
    
% Load recording 
    
    tmp = load(tmpFile,"-mat");
    data = tmp.Data;

% Convert the raw bern labels to labels and resample to 200 ms

    rawLabels = [data.labels_O1, data.labels_O2]';
    preLabels = convert_labels_to_classes(rawLabels);

    % Append to table
    tmpTable = struct2table(flipStruct(preLabels));
    tmpTable.id = repelem(tmpId, length(preLabels.W), 1);
    rawTable = [rawTable; tmpTable];

    postLabels = structfun(@(s) apply_moving_func(s, ...       % label
                                                       data.fs, ... % fs
                                                       params.windowSizeSec, ... 
                                                       params.windowStepSec, ...     
                                                       params.downsampleFcn), ... 
                                preLabels, 'UniformOutput', false);
    
    % Append to table
    tmpTable = struct2table(flipStruct(postLabels));
    tmpTable.id = repelem(tmpId, length(postLabels.W), 1);
    postTable = [postTable; tmpTable];

    % Only bilateral-wake vs bilateral MSE
    biWake_vs_biMSE = get_grouped_classes(postLabels.W, postLabels.MSE);
    
    % Wake + ED + MSEc + MSEu vs bilateral MSE
    rest_vs_biMSE = get_grouped_classes([postLabels.W; postLabels.ED; ...       %neg
                                        postLabels.MSEc; postLabels.MSEu], ...  
                                        postLabels.MSE);                    %pos
    
    % Wake vs bilateral-MSE + ED + MSEc + MSEu
    biWake_vs_rest = get_grouped_classes(postLabels.W, ...                  % neg
                                         [postLabels.MSE; postLabels.MSEu; ...  % pos
                                          postLabels.MSEc; postLabels.ED]);

    % Wake + ED vs bilateral-MSE + MSEu + MSEc
    wakeED_vs_restMSE = get_grouped_classes([postLabels.W; postLabels.ED], ...    %neg
                                            [postLabels.MSE; postLabels.MSEu; ... %pos
                                            postLabels.MSEc]); 

     % Append to table
    tmpTable = table;
    tmpTable.id = repelem(tmpId, length(biWake_vs_rest), 1);
    tmpTable.train = repelem(ismember(tmpId, params.split.train), length(biWake_vs_biMSE), 1);
    tmpTable.biWake_vs_biMSE = biWake_vs_biMSE';
    tmpTable.rest_vs_biMSE = rest_vs_biMSE';
    tmpTable.biWake_vs_rest = biWake_vs_rest';
    tmpTable.wakeED_vs_restMSE = wakeED_vs_restMSE';
    grpTable = [grpTable; tmpTable];

end

%% 

vars = string(rawTable.Properties.VariableNames(~ismember(rawTable.Properties.VariableNames,["W","id"])));

rawDurs = varfun(@(x) {get_durations(x, 1/200)}, rawTable, ...
                "InputVariables", vars, ...
                "GroupingVariables","id");
%% Raw stats
rawStats = [];
for i = 1:numel(vars)
    var = vars(i);
    funVar = "Fun_" + var;
    tmpStats = calc_stats(cell2mat(rawDurs.(funVar)'));
    tmpStats.subCount = sum(cellfun(@(x) numel(x) > 1, rawDurs{:,funVar}));
    tmpStats.varPerc = sum(rawTable.(var)) / height(rawTable);
    rawStats = [rawStats; tmpStats];
end
rawStatsTable = struct2table(rawStats);
rawStatsTable = rows2vars(rawStatsTable);
rawStatsTable.Properties.VariableNames = ["s", vars];

%% Group stats
vars = "wakeED_vs_restMSE";
funVar = "Fun_"+vars;
grpDurs = varfun(@(x) {get_durations(x, params.windowSizeSec)}, grpTable, ...
                "InputVariables", vars, ...
                "GroupingVariables",["id","train"]);

%%
% Overall
overallGrpStats = calc_stats(cell2mat(grpDurs.(funVar)'));
overallGrpStats.subCount =  sum(cellfun(@(x) numel(x) > 1, grpDurs{:,funVar}));
overallGrpStats.varPerc = sum(grpTable.(vars) == 1) / height(grpTable);


% Training only
trainGrpDurs = grpDurs(grpDurs.train,:);
trainGrpStats = calc_stats(cell2mat(trainGrpDurs.(funVar)'));
trainGrpStats.subCount =  sum(cellfun(@(x) numel(x) > 1, trainGrpDurs{:,funVar}));
trainGrpStats.varPerc = sum(grpTable{grpTable.train,vars}==1) / sum(grpTable.train);

% Test only
testGrpDurs = grpDurs(~grpDurs.train,:);
testGrpStats = calc_stats(cell2mat(testGrpDurs.(funVar)'));
testGrpStats.subCount =  sum(cellfun(@(x) numel(x) > 1, testGrpDurs{:,funVar}));
testGrpStats.varPerc = sum(grpTable{~grpTable.train,vars}==1) / sum(~grpTable.train);

grpStatsTable = struct2table([overallGrpStats, trainGrpStats, testGrpStats]);
grpStatsTable = rows2vars(grpStatsTable);
grpStatsTable.Properties.VariableNames = ["s", "Overall", "Train", "Test"];


%%
labelStatsTable = join(rawStatsTable, grpStatsTable, "Keys", "s")

%% Make label stats for training validation and test splits
valSplits = parse_json("..\splits\skorucack_splits_with_validation.json");
gTable = grpTable;
gTable.type = strings(size(gTable,1),1);
gTable.type(ismember(gTable.id, valSplits.train)) = "train";
gTable.type(ismember(gTable.id, valSplits.valid)) = "valid";
gTable.type(ismember(gTable.id, valSplits.test)) = "test";


vars = "wakeED_vs_restMSE";
funVar = "Fun_"+vars;
gDurs = varfun(@(x) {get_durations(x, params.windowSizeSec)}, gTable, ...
                "InputVariables", vars, ...
                "GroupingVariables",["id","type"]);

%%

% Training only

trainGrpDurs = gDurs(gDurs.type=="train",:);
trainGrpStats = calc_stats(cell2mat(trainGrpDurs.(funVar)'));
trainGrpStats.subCount =  sum(cellfun(@(x) numel(x) > 1, trainGrpDurs{:,funVar}));
trainGrpStats.varPerc = sum(gTable{gTable.type=="train",vars}==1) / sum(gTable.type=="train");

% Valid only
validGrpDurs = gDurs(gDurs.type=="valid",:);
validGrpStats = calc_stats(cell2mat(validGrpDurs.(funVar)'));
validGrpStats.subCount =  sum(cellfun(@(x) numel(x) > 1, validGrpDurs{:,funVar}));
validGrpStats.varPerc = sum(gTable{gTable.type=="valid",vars}==1) / sum(gTable.type=="valid");


% Test only
testGrpDurs = gDurs(gDurs.type=="test",:);
testGrpStats = calc_stats(cell2mat(testGrpDurs.(funVar)'));
testGrpStats.subCount =  sum(cellfun(@(x) numel(x) > 1, testGrpDurs{:,funVar}));
testGrpStats.varPerc = sum(gTable{gTable.type=="test",vars}==1) / sum(gTable.type=="test");

gStatsTable = struct2table([trainGrpStats, validGrpStats, testGrpStats]);
gStatsTable = rows2vars(gStatsTable);
gStatsTable.Properties.VariableNames = ["s", "Train", "Validation", "Test"]





