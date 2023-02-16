% close all; clear all; clc;

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
    hdr.SignalLabels        = signalLabels;
    hdr.PhysicalDimensions  = repelem("uV", numSignals);
    hdr.PhysicalMin         = int16(repelem(-3000, numSignals));
    hdr.PhysicalMax         = int16(repelem(3000, numSignals));
    hdr.DigitalMin          = int16(repelem(-8192, numSignals));
    hdr.DigitalMax          = int16(repelem(8192, numSignals));
    hdr.Prefilter           = repelem("BP 0.5-45 Hz.", numSignals);

    % Write edf
    subdir = fullfile("C:\Users\SIOS\U-Time\transfer_learning\data", hdr.Patient);
    edfFile = fullfile(subdir, sprintf("%s_psg.edf",hdr.Patient));

    targets = master{master.id==hdr.Patient, params.testTargets};

    %
    tsal1 = timetable;
    if any(targets==1)


        % Sleep
        [f,l,s] = get_first_and_last(targets,1);

        f=f-1;
        onset = seconds(f');
        annotations = string(ones(size(f')));
        duration = seconds(((l-f)))';

        if ~isempty(s)
            onset = [onset; seconds((s-1)')];
            annotations = [annotations; string(ones(size(s')))];
            duration = [duration; seconds(ones(size(s')))];
        end

        tsal1=timetable(onset,annotations,duration);

    end

    % Wake
    tt=targets~=1;
    [f,l,s] = get_first_and_last(tt,1);
    f=f-1;
    onset = seconds(f');
    annotations = string(zeros(size(f')));
    duration = seconds(((l-f)))';

    if ~isempty(s)
        onset = [onset; seconds((s-1)')];
        annotations = [annotations; string(ones(size(s')))];
        duration = [duration; seconds(ones(size(s')))];
    end


    tsal2 = timetable(onset,annotations,duration);
    tsal = sortrows([tsal1; tsal2]);


    if ~isempty(tsal1)
        if overlapsrange(tsal1, tsal2)
            fprintf("%s: Overlap!\n",hdr.Patient);
        end
    end
    if strcmpi(hdr.Patient,"0ncr")
        1+1;
    end


    if ismember(hdr.Patient, params.split.train)
        mkdir(subdir)
        edfw = edfwrite(edfFile,hdr,signals,tsal1);
        edfw = edfwrite(edfFile.replace("_psg","_hyp"), hdr, signals, tsal);
    end


    %     % Write out labels
    %     labels = struct;
    %     labels.O1 = data.labels_O1;
    %     labels.O2 = data.labels_O2;
    %     labelsFile = strcat("C:\code\master_thesis\labels\", hdr.Patient, ".mat");
    %     save(labelsFile, "labels",'-mat');

end

if barOn
    waitbar(1, wbar, "DONE!");
    pause(0.5);
    allWbr = findall(0,'type','figure','tag','TMWWaitbar');
    delete(wbar);
end