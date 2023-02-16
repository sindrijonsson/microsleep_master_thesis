function [processedPredictions] = predict_and_post_process(X, model, asCell)

if nargin < 3; asCell = 0; end

yHatCategorical = categorical(predict(model, X));
yHatRaw = grp2idx(yHatCategorical)-1;
yHatRaw(yHatRaw==0) = -1;

% Apply post-processing

% 1. Remove MS predictions shorter than 1 seconds
fixedYhat = remove_invalid_labels(yHatRaw, 1);

% 2. Apply 9s moving median filter to RFmodel and SVMmodel predictions
yHat = movmedian(fixedYhat,45,"Endpoints","shrink");
yHat(yHat==0) = 1;

if asCell
    processedPredictions = num2cell(yHat,1);
else
    processedPredictions = yHat;
end

end