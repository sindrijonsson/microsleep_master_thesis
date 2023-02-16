function features = generate_features(data, params)

% Defaults
fs = data.fs;
order = 16;
    
if nargin < 2
    windowSizeSec = 1;
    windowStepSec = 0.2;
    df = 0.2;
    zeroPad = [];
else
    windowSizeSec = params.pburgWindowSize;
    windowStepSec = params.windowStepSec;
    df = params.pburgFreqRes;
    zeroPad = params.zeroPad;
end
    
% Get signals and set to [1xnum_samples]
o1 = data.eeg_O1';
o2 = data.eeg_O2';

delta_e = data.E1 - data.E2;
delta_e = delta_e';

if ~isempty(zeroPad)
    o1 = [zeroPad, o1, zeroPad];
    o2 = [zeroPad, o2, zeroPad];
    delta_e = [zeroPad, delta_e, zeroPad];
end

occipital_channels = [o1; o2];



% Calculate power spectrum for delta EOG

[PxxEOG, ~, ~] = moving_pburg(delta_e, ...
    order, ...              % AR Order
    fs, ...                 % Sampling frequency [Hz]
    windowSizeSec, ...    % Window size [s]
    windowStepSec, ...    % Step size [s]
    df ...                  % Frequency Resolution [Hz]
    );

features = zeros(size(PxxEOG,2),14);

% Calculate features for occipital channels
for i = 1:height(occipital_channels)

    y = occipital_channels(i,:);    % Get occipital channel
    fprintf("Generating features for O%i\n",i)

    [Pxx, f, ~] = moving_pburg(y, ...
        order, ...              % AR Order
        fs, ...                 % Sampling frequency [Hz]
        windowSizeSec, ...      % Window size [s]
        windowStepSec, ...      % Step size [s]
        df ...                  % Frequency Resolution [Hz]
        );


    % Get powerband indexes
    idx = get_band_indexes(f, 0);

    % Apply median filter to power bands
    fcn = @(x) movmedian(x, 1/windowStepSec, 2, "Endpoints", "shrink");

    % Calculate powerbands
    delta = fcn(sum(Pxx(idx.delta,:)) * df);
    theta = fcn(sum(Pxx(idx.theta,:)) * df);
    alpha = fcn(sum(Pxx(idx.alpha,:)) * df);
    beta  = fcn(sum(Pxx(idx.beta,:)) * df);

    % Calculate theta - alpha beta ratioe (T/(A+B))
    TAB = theta ./ (alpha + beta);

    % Calculate eye movement estimate
    delta_EOG = sum(PxxEOG(idx.delta,:)) * df;
    eyeMovement = delta_EOG ./ delta;

    % Calculate median frequency
    Pxx_normalized = Pxx(idx.total,:) ./ sum(Pxx(idx.total,:));
    Pxx_cumulative = cumsum(Pxx_normalized);
    [~, medIdx] = min(abs(Pxx_cumulative-0.5));
    totalFreq = f(idx.total);
    medFreq = totalFreq(medIdx)';

    % Store features as num_windows x 7
    tmpFeatures = [delta;
        theta;
        alpha;
        beta;
        TAB;
        eyeMovement;
        medFreq]';

    tmpNumFeatures = size(tmpFeatures,2);
    featureIdx = (1:tmpNumFeatures)+(tmpNumFeatures*(i-1));
    features(:,featureIdx) = tmpFeatures;

end
fprintf("[%ix%i] Features succesfully generated\n", ...
        size(features,1), size(features,2));

end

%%
function idx = get_band_indexes(f, print)

% Indexes of power bands
f = round(f,1);

if nargin < 2; print = 1; end

idx=struct;
idx.delta = (f >= 0.8)  &   (f < 4);       % Delta: 0.8 - 4 Hz
idx.theta = (f >= 4)    &   (f < 8);       % Theta: 4 - 8 Hz
idx.alpha = (f >= 8)    &   (f < 12);      % Alpha: 8 - 12 Hz
idx.beta  = (f >= 12)   &   (f <= 26);     % Beta:  12 - 26 Hz
idx.total = (f >= 0.8)  &   (f <= 26);     % Total: 0.8 - 26 Hz

if print
    fn = fieldnames(idx);
    fprintf("\nBands\n\n")
    for j = 1:numel(fn)
        band = fn(j);
        freqs = f(idx.(band{1}));
        fprintf("%s:\t[%.1f - %.1f]\n", string(band), freqs(1), freqs(end));
    end
end
end
