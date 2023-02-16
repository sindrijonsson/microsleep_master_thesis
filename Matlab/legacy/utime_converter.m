close all; clear all; clc;

%% Constants

signalVariables = ["eeg_O1", "eeg_O2", "E1", "E2"];
signalType = ["EEG", "EEG", "EOG", "EOG"];
signalLabels = ["O1-M2", "O2-M1", "E1-M1", "E2-M1"];
% signalNames = strcat(signalType, " ", signalLabels);


%% Get all files

files = struct2table(dir("data\"));
files = files(files.bytes > 0,:);

%% Load tmp file
barOn = 0;

if barOn
    wbar = waitbar(0, "Processing");
end

% for i = 1:height(files)
i = 14
    tmpFile = string(files.name(i));
    tmpPath = strcat("data\", tmpFile);
    if barOn
        waitbar(i/height(files), wbar, sprintf("Processing: %s",tmpFile))
    end
    out = load(tmpPath);

    data = out.Data;
    fs = data.fs;

    % Collect signals
    fn = fieldnames(data);
    signalFields = find(ismember(fn,signalVariables));
    numSignals = numel(signalFields);

    assert(numSignals == length(signalVariables))

    signals = zeros(data.num_Labels, numSignals);
    for i = 1:numel(signalFields)
        signals(:,i) = data.(fn{signalFields(i)});
    end

    % Create header
    hdr = edfheader("EDF");

    hdr.Patient             = replace(tmpFile,".mat","");
    hdr.Recording           = tmpFile;
    hdr.StartDate           = "12.12.12";
    hdr.StartTime           = "00.00.00";
    hdr.NumDataRecords      = 1;
    hdr.DataRecordDuration  = seconds(data.num_Labels/data.fs);
    hdr.NumSignals          = numSignals;
    hdr.SignalLabels        = signalLabels;
    hdr.PhysicalDimensions  = repelem("uV", numSignals);
    hdr.PhysicalMin         = int16(repelem(-3000, numSignals));
    hdr.PhysicalMax         = int16(repelem(3000, numSignals));
    hdr.DigitalMin          = int16(repelem(-8192, numSignals));
    hdr.DigitalMax          = int16(repelem(8192, numSignals));
    hdr.Prefilter           = repelem("BP 0.5-45 Hz.", numSignals);


    % Make labels
    labels = struct;
    labels.O1 = data.labels_O1;
    labels.O2 = data.labels_O2;
    
    bernLabels = [labels.O1';
                  labels.O2'];

    bilateralMS = all(bernLabels==1)*1;

%     out = get_start_and_stop(bilateralMS,1);
%     onset = seconds(out.start / fs);
%     offset = seconds(out.stop / fs);
    onset = seconds(0:1/fs:((length(bilateralMS)-1)/fs))
    ;
    annotations = strings(size(bilateralMS));
    annotations(bilateralMS==0) = "0";
    annotations(bilateralMS==1) = "1";
    duration = seconds(ones(size(bilateralMS))/fs);
    annotationsList = timetable(onset',annotations',duration');


    % Write edf
    edfFile = strcat("C:\code\U-Time\Bern\", hdr.Patient,".edf");
    edfw = edfwrite(edfFile,hdr,signals,annotationsList);


%%

%     labelsFile = strcat("C:\code\U-Time\Bern\", hdr.Patient, ".mat");
%     save(labelsFile, "labels",'-mat');

% end
% 
% if barOn
% waitbar(1, wbar, "DONE!");
% pause(0.5);
% % allWbr = findall(0,'type','figure','tag','TMWWaitbar');
% delete(wbar);
% end

function out=get_start_and_stop(labels, target)
    idx = find(labels == target);
    D = diff([0,diff(idx)==1, 0]);
    start = idx(D>0);
    stop = idx(D<0);
    out = struct;
    out.start = start;
    out.stop = stop;
end
