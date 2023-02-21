% F1 Matrix

%% Gather all test data in one table

transUsleepTest = table;
transUsleepTest.id = testUsleep.id;
transUsleepTest.yTrue = cellfun(@(x) num2cell(x',1), testUsleep.yTrue);
transUsleepTest.yHat = cellfun(@(x) num2cell(x',1), testUsleep.yHat);

transmUsleepTest = table;
transmUsleepTest.id = testMuSleep.id;
transmUsleepTest.yTrue = cellfun(@(x) num2cell(x',1), testMuSleep.yTrue);
transmUsleepTest.yHat = cellfun(@(x) num2cell(x',1), testMuSleep.yHat);

transMalafeevTest = table;
transMalafeevTest.id = testMalafeev.id;
transMalafeevTest.yTrue = cellfun(@(x) num2cell(x',1), testMalafeev.yTrue);
transMalafeevTest.yHat = cellfun(@(x) num2cell(x',1), testMalafeev.yHat);

transSSLTest = table;
transSSLTest.id = testSSL.id;
transSSLTest.yTrue = cellfun(@(x) num2cell(x',1), testSSL.yTrue);
transSSLTest.yHat = cellfun(@(x) num2cell(x',1), testSSL.yHat);

expert = table;
expert.id = testLSTM.id;
expert.yHat = testLSTM.yTrue;
expert.yTrue = testLSTM.yTrue;

% Join test results into common table
allTest = {transmUsleepTest;
    transUsleepTest;
    transSSLTest;
    testLSTM;
    testRF;
    testSVM;
    transMalafeevTest;
    expert};

models = [mdls; "Human"]

%%

% Columns actual
% Rows predicted

m = zeros(size(allTest,1));

var = "f1";

for c = 1:numel(allTest)
    ref = allTest{c};
    for r = 1:numel(allTest)
        tmp = allTest{r};
        test = table;
        test.yTrue = ref.yHat;
        test.yHat = tmp.yHat;
        test.id = ref.id;
        [stats, ~] = eval_test_by(test, "sample", params);
        m(r,c) = stats.(var); 
        if r==c; m(r,c) = nan; end
    end
end

%% Plot
colormap default
figure(1000);
ax=imagesc(m); hold on;

t = cellfun(@(x) sprintf("%.2f",x), num2cell(m), 'UniformOutput', false); % convert to string

s = 1:numel(models)+1;
n = numel(models);
for i = 1:numel(s)
    xline(s(i)-0.5,"k-")
    yline(s(i)-0.5,'k-')
end
x=repmat(1:n,n,1);
y=x';
text(x(:), y(:), t, 'HorizontalAlignment','center',"FontSize",12,"FontWeight","bold")
colorbar eastoutside
title("F1 - Score Confusion Matrix")
% caxis([0, 1])

xticklabels(models)
yticklabels(models)