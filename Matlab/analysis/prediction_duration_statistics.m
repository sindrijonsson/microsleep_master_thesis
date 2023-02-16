tests = {testMuSleep, testUsleep, testLSTM, testRF, testSVM, malafeev};

m = [];
d = [];
for i = 1:numel(tests)
    tmp = tests{i};
    durs = cellfun(@(x) get_durations(x, params.secPerLabel), ...
                    tmp.yHat, ...
                    "UniformOutput",false);
    durs = cell2mat(durs');
    m = [m, repmat(mdls(i), size(durs))];
    d = [d, durs];
end
t = table;
t.model = m';
t.durs = d';
%%

figure(100); clf; hold on;
for i = 1:numel(mdls)
    boxplot(d(m==mdls(i)),"Positions",i);
end
xticks(1:numel(mdls))
xticklabels(mdls)

%%
figure(200); clf; hold on;
for i = 1:numel(mdls)
    histogram(d(m==mdls(i)),200);
    xlim([0,50])
end
legend(mdls)

%%
figure(300); clf; hold on;
for i = 1:numel(mdls)
    ys = d(m==mdls(i));
    xs = ones(size(ys)) * i*2;
    scatter(ys,xs,".","YJitter","density");
end

legend(mdls,"Interpreter","none")
