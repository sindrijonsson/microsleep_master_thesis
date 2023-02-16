close all
clear all
clc

% Get all the files

list = string(ls("data"));
list = list(arrayfun(@(x) ~startsWith(x,"."), list));
names = replace(list,".mat","");
fullPaths = strcat("data\", list);

% --------------------------------------------------------------------------

% Read the train and test split

fid = fopen("..\\splits\\skorucack_splits.json"); 
raw = fread(fid,inf); 
str = char(raw'); 
fclose(fid); 
splits = jsondecode(str);

% -------------------------------------------------------------------------

dataTable = table;

% Iterate through files and generate master data

for fi = 1:numel(fullPaths)

    tmpFile = fullPaths(fi);
    tmpName = names(fi);
    
    fprintf("Processing %s\n",tmpName);

% Load recording 
    
    tmp = load(tmpFile,"-mat");
    data = tmp.Data;

% -------------------------------------------------------------------------

% Calculate features
    
    features = generate_features(data, 1);
    feature_cell = num2cell(features,2);    % Easy storage

% -------------------------------------------------------------------------
% Convert the raw bern labels to labels and resample to 200 ms
    fprintf("Converting labels\n");

    bern_labels = [data.labels_O1, data.labels_O2]';
    labels = convert_labels_to_classes(bern_labels);
    
    resampledLabels = structfun(@(s) apply_moving_func(s, ...       % label
                                                       data.fs, ... % fs
                                                       0.2, ...     % win size [s]
                                                       0.2, ...     % win step [s]
                                                       @(x) median(x,2)), ... %func
                                labels, 'UniformOutput', false);


% -------------------------------------------------------------------------

% Group the labels as in Skorucack 2020
%   -1 => Wake 
%    1 => MS
    
    % Only bilateral-wake vs bilateral MSE
    biWake_vs_biMSE = get_grouped_classes(resampledLabels.W, resampledLabels.MSE);
    
    % Wake + ED + MSEc + MSEu vs bilateral MSE
    rest_vs_biMSE = get_grouped_classes([resampledLabels.W; resampledLabels.ED; ...       %neg
                                        resampledLabels.MSEc; resampledLabels.MSEu], ...  
                                        resampledLabels.MSE);                    %pos
    
    % Wake vs bilateral-MSE + ED + MSEc + MSEu
    biWake_vs_rest = get_grouped_classes(resampledLabels.W, ...                  % neg
                                         [resampledLabels.MSE; resampledLabels.MSEu; ...  % pos
                                          resampledLabels.MSEc; resampledLabels.ED]);
    
    % Wake + ED vs bilateral-MSE + MSEu + MSEc
    wakeED_vs_restMSE = get_grouped_classes([resampledLabels.W; resampledLabels.ED], ...    %neg
                                            [resampledLabels.MSE; resampledLabels.MSEu; ... %pos
                                            resampledLabels.MSEc]); 

% -------------------------------------------------------------------------

% Collect information and append to table 

    recording = struct;
    % Info
    sz=size(feature_cell);
    recording.patient = repmat(tmpName, sz);
    recording.train = repmat(ismember(tmpName, splits.train), sz);
    recording.sample = [1:size(feature_cell,1)]';
    
    % Features and labels
    recording.features = feature_cell;
    recording.biWake_vs_biMSE = biWake_vs_biMSE';
    recording.rest_vs_biMSE = rest_vs_biMSE';
    recording.biWake_vs_rest = biWake_vs_rest';
    recording.wakeED_vs_restMSE = wakeED_vs_restMSE';

    recordingTable = struct2table(recording);

    posStats = sum(table2array(recordingTable(:,5:8))==1,"omitnan");
    negStats = sum(table2array(recordingTable(:,5:8))==-1,"omitnan");
    
    fprintf("\t\t\t MS \t Wake\n")
    for j = 1:numel(posStats)
        fprintf("Grouping %i: %i \t %i \n",j,posStats(j),negStats(j));
    end

    
    
    dataTable = [dataTable; recordingTable];

    if (mod(fi, 5) == 0) | fi == height(fullPaths)
        fprintf("Saving table at checkpoint %i...\n\n",fi);
        last_idx = fi;
        checkPoint = struct;
        checkPoint.dataTable = dataTable;
        checkPoint.last_idx = fi;
        save("master_zeropad.mat", "checkPoint");
    end

end