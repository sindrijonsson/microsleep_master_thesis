tests = {testMuSleep, testUsleep, testLSTM, testRF, testSVM, malafeev};
tt = table;
for i = 1:numel(tests)
    tmp = tests{i};
    try 
        durs = get_durations(cell2mat(tmp.yHat'), 1);
    catch
        durs = get_durations(cell2mat(tmp.yHat), 1);
    end
    t = table;
    t.i = repmat(i,size(durs,2),1);
    t.val = durs';
    tt = [tt; t]
end

figure(2); clf;
boxplot(tt.val, tt.i)
xticklabels(mdls)