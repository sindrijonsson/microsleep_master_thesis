
function master = init_master(params)

% ------------------------------------------------------------------------
filePaths = params.filePaths;
fileIds = params.fileIds;
split = params.split;

% Sampling values
winSizeSec = params.windowSizeSec;
winStepSec = params.windowStepSec;
downSampleFcn = params.downsampleFcn;

% -------------------------------------------------------------------------

% -------------------------------------------------------------------------


if exist(params.masterFile)
    inp = input("Warning! Master exists, do you want to overwrite [y/n]?: ", ...
                's');
    if ~strcmp(inp,"y")
        master = load(params.masterFile);
        master = master.checkPoint.master;
        return
    end
end


master = table;

% Iterate through files and generate master data

for fIdx = 1:numel(filePaths)

    tmpFile = filePaths(fIdx);
    tmpId = fileIds(fIdx);
    
    fprintf("Processing %s\n", tmpId);

% Load recording 
    
    tmp = load(tmpFile,"-mat");
    data = tmp.Data;

% -------------------------------------------------------------------------

% Calculate features
    
    features = generate_features(data, params);
    featureCell = num2cell(features, 2);    % Easy storage

% -------------------------------------------------------------------------
% Convert the raw bern labels to labels and resample to 200 ms
    fprintf("Converting labels\n");

    bern_labels = [data.labels_O1, data.labels_O2]';
    labels = convert_labels_to_classes(bern_labels);
    
    resampledLabels = structfun(@(s) apply_moving_func(s, ...       % label
                                                       data.fs, ... % fs
                                                       winSizeSec, ... 
                                                       winStepSec, ...     
                                                       downSampleFcn), ... 
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
    recording.id = repmat(tmpId, size(featureCell));
    recording.train = repmat(ismember(tmpId, split.train), size(featureCell));
    recording.sample = [1:size(featureCell,1)]';
    
    % Features and labels
    recording.features = featureCell;
    recording.biWake_vs_biMSE = biWake_vs_biMSE';
    recording.rest_vs_biMSE = rest_vs_biMSE';
    recording.biWake_vs_rest = biWake_vs_rest';
    recording.wakeED_vs_restMSE = wakeED_vs_restMSE';

    master = [master; struct2table(recording)];

    if (mod(fIdx,5) == 0) || (fIdx==height(filePaths))
        fprintf("Saving table...\n\n");
        checkPoint = struct;
        checkPoint.master = master;
        checkPoint.last_idx = fIdx;
        save(params.masterFile, "checkPoint");
    end
    
    
end