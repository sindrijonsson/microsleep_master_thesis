function metrics = calc_performance_metrics(y_true, y_hat, params, order)

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

        % Recall = TP / (TP + FN)
        if (tp + fn) > 0
            metrics.recall = tp / (tp + fn);
        else
            metrics.recall = 0;
        end
        
        % Precision = TP / (TP + FP)
        
        if (tp + fp) > 0
            metrics.precision = tp / (tp + fp);
        else
            metrics.precision = 0;
        end
        
        % F1-Score = (2 x Precision x Recall) / (Precision + Recall)
        if (metrics.precision + metrics.recall) > 0
            metrics.f1 = (2* metrics.precision * metrics.recall) / ...
                         (metrics.precision + metrics.recall);
        else
            metrics.f1 = 0;
        end
%         metrics.kappa = calc_kappa(cm);
            
    else
        metrics.recall = nan;
        metrics.precision = nan;
        metrics.f1 = nan;
    end
end