tests = {testMuSleep, testUsleep, testSSL, testLSTM, testRF, testSVM, testMalafeev};

tSamp = table;
tSampPerRec = table;

tEvent = table;
tEventPerRec = table;

for i = 1:numel(tests)
    tmp = tests{i};
    tmp.yTrue = cellfun(@(x) remove_invalid_labels(x, 3, 15, 1, nan, false), ...
                        tmp.yTrue, ...
                        "UniformOutput",false);
    tmp.yHat = cellfun(@(x) remove_invalid_labels(x, 3, 15, 1, nan, false), ...
                      tmp.yHat, ...
                      "UniformOutput",false);

    [outSamp, perRecSamp] = eval_test_by(tmp, "sample", params);
    outSampPerRec = pretty_per_rec(perRecSamp, "sample");

    tSamp = [tSamp; struct2table(outSamp)];
    tSampPerRec = [tSampPerRec; struct2table(outSampPerRec)];
    
    [outEvent, perRecEvent] = eval_test_by(tmp, "event", params);
    outEventPerRec = pretty_per_rec(perRecEvent, "event");

    tEvent = [tEvent; struct2table(outEvent)];
    tEventPerRec = [tEventPerRec; struct2table(outEventPerRec)];

end

tSamp.models = mdls; 
tSamp = tSamp(:,[size(tSamp,2),(1:end-1)])

tSampPerRec.models = mdls;
tSampPerRec = tSampPerRec(:,[end,(1:end-1)])

tEvent.models = mdls;
tEvent = tEvent(:,[size(tEvent,2),(1:end-1)])

tEventPerRec.models = mdls;
tEventPerRec = tEventPerRec(:,[size(tEventPerRec,2),(1:end-1)])

% write_results_table(tSamp, tSampPerRec, "sample", params, "alt");
% write_results_table(tEvent, tEventPerRec, "event", params, "alt");

function entry = pretty_per_rec(perRec, by)
    [m, s] = summarize_perRecMetrics(perRec, by);
    info = cellfun(@(m,s) sprintf("%.2f +/- %.2f",m,s), struct2cell(m),struct2cell(s));
    entry = cell2struct(cellstr(info),fieldnames(m));
end