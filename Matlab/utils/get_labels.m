function out = get_labels(file, params)


% Load recording

tmp = load(file,"-mat");
data = tmp.Data;

% Convert the raw bern labels to labels and resample to 200 ms
% fprintf("Converting labels\n");

bern_labels = [data.labels_O1, data.labels_O2]';
out = convert_labels_to_classes(bern_labels);

if nargin < 2
    return
else
    out = structfun(@(s) apply_moving_func(s, ...       % label
        data.fs, ... % fs
        params.windowSizeSec, ...
        params.windowStepSec, ...
        params.downsampleFcn), ...
        out, 'UniformOutput', false);
end

end