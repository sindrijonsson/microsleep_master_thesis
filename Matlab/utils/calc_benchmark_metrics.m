function metrics = calc_benchmark_metrics(y_true, y_hat, params, order)
if nargin < 3; timePerPrediction = 0.2; 
    else; timePerPrediction = params.windowSizeSec; end
    if nargin < 4; order = [1, -1]; end


    metrics = struct;

    % Generate confusion matrix
    cm = confusionmat(y_true, y_hat, "Order", order);
    % +----+----+
    % | TP | FN |
    % |----+----|
    % | FP | TN |
    % +----+----+
    tp = cm(1,1);
    fn = cm(1,2);
    fp = cm(2,1);
    tn = cm(2,2);
    
   
    % Specificity = TN / (FP + TN)
    metrics.specificity = tn / (fp + tn);
   
    % Accuracy = (TP + TN) / (TP + TN + FP + FN)
    metrics.accuracy = (tp+tn) / sum(cm,"all");

    % False positives in minutes
    metrics.fpMinutes = minutes( seconds(fp*timePerPrediction) );


    % Check if there are any positive classes
    [f,~,s] = get_first_and_last(y_true, 1, 0);
    is_pos = (length(f) + length(s) ) > 1;

    % If there are then calculate sensitivity, precision and Cohen's kappa
    if is_pos

        % Sensitivity (recall) = TP / (TP + FN)
        metrics.recall = tp / (tp + fn);

        % Precision = TP / (TP + FP)
        metrics.precision = tp / (tp + fp);

        % Cohen's Kappa = (po - pe) / (1 - pe)
        metrics.kappa = calc_kappa(cm);
    
    % Otherwise set as nan (to omit in mean calculation)
    else
        metrics.recall = nan;
        metrics.precision = nan;
        metrics.kappa = nan;
    end

end