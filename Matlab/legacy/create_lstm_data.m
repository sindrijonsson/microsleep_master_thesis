function data = create_lstm_data(features, name)

if nargin < 2; name = ""; end
fprintf("Processing %s\n",unique(name));

numFeatures = 14;
winSizeSamples = 45;
numWins = height(features) - (winSizeSamples-1);
winMask = repmat(1:45,numWins,1);
winStep = 1 * (0:numWins-1)';
windows = winMask+winStep;

middleSeqX = cellfun(@(x) mat2cell(vertcat(features{x})', numFeatures, size(x,2)), ...
                    num2cell(windows,2));

startPadIdx = generate_padding_indexes(winSizeSamples);
endPadIdx = (height(features)+1)-flip(startPadIdx,2); 
endPadIdx(endPadIdx>height(features)) = 0;

startPadSeqX = cellfun(@(x) mat2cell(extract_and_pad(features, x, "left"), ...
                                     numFeatures, size(x,2)), ...
                       num2cell(startPadIdx,2));

endPadSeqX = cellfun(@(x) mat2cell(extract_and_pad(features, x, "right"), ...
                                   numFeatures, size(x,2)), ...
                    num2cell(endPadIdx,2));

% Concat data
data = [startPadSeqX;
        middleSeqX;
        endPadSeqX];

end


function out = generate_padding_indexes(winSizeSamples)

    outsideWindow = (winSizeSamples-1)/2;
    samples = 1:winSizeSamples;
    out = nan(outsideWindow,winSizeSamples);
    for i = 1:outsideWindow
        numPads = (outsideWindow+1)-i;
        numPrev = i - 1;
        if numPrev; idxPrev = 1:numPrev; else; idxPrev = []; end
        idxNext = i:i+outsideWindow;
        out(i,:)=[zeros(1,numPads),samples(idxPrev),samples(idxNext)];
    end

end


function out = extract_and_pad(in, idx, side)
    pad_idx = idx==0;
    numFeatures = size(in{1},2);
    z = zeros(numFeatures,sum(pad_idx));
    x = vertcat(in{idx(~pad_idx)})';
    if side == "left"; out = [z, x]; 
    elseif side == "right"; out = [x, z]; end
end
