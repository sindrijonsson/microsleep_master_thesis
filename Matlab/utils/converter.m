close all; clear all; clc;

%% Constants

signalVariables = ["eeg_O1", "eeg_O2", "E1", "E2"];
signalType = ["EEG", "EEG", "EOG", "EOG"];
signalLabels = ["O1-M2", "O2-M1", "LOC-M1", "ROC-M1"];
signalNames = strcat(signalType, " ", signalLabels);


%% Get all files

files = struct2table(dir("data\"));
files = files(files.bytes > 0,:);

%% Load tmp file
barOn = 1;

if barOn
    wbar = waitbar(0, "Processing");
end

for i = 1:height(files)

    tmpFile = string(files.name(i));
    tmpPath = strcat("data\", tmpFile);
    if barOn
        waitbar(i/height(files), wbar, sprintf("Processing: %s",tmpFile))
    end
    out = load(tmpPath);

    data = out.Data;


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
    hdr.SignalLabels        = signalNames;
    hdr.PhysicalDimensions  = repelem("uV", numSignals);
    hdr.PhysicalMin         = int16(repelem(-3000, numSignals));
    hdr.PhysicalMax         = int16(repelem(3000, numSignals));
    hdr.DigitalMin          = int16(repelem(-8192, numSignals));
    hdr.DigitalMax          = int16(repelem(8192, numSignals));
    hdr.Prefilter           = repelem("BP 0.5-45 Hz.", numSignals);

    % Write edf
    edfFile = strcat("C:\code\master_thesis\edf_data", hdr.Patient,".edf");
    edfw = edfwrite(edfFile,hdr,signals);

    % Write out labels
    labels = struct;
    labels.O1 = data.labels_O1;
    labels.O2 = data.labels_O2;
    labelsFile = strcat("C:\code\master_thesis\labels\", hdr.Patient, ".mat");
    save(labelsFile, "labels",'-mat');

end

if barOn
waitbar(1, wbar, "DONE!");
pause(0.5);
% allWbr = findall(0,'type','figure','tag','TMWWaitbar');
delete(wbar);
end