function out = convert_labels_to_only_bilateral_mse_vs_wakefulness(y)
    
    out = nan(1,size(y,2));
    bilateralMSE = all(y == 1);
    bilateralWake = all(y == 0);
    
    out(bilateralWake) = 0;
    out(bilateralMSE) = 1;


end