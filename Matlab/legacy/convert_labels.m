function out = convert_labels(y, wake, ms)

    wakeIdx = ismember(y, wake);
    msIdx = ismember(y, ms);
    
    out = zeros(size(y));
    out = all(msIdx);

end