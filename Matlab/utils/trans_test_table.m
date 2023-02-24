function out = trans_test_table(tbl, probs)

out = table;
out.id = tbl.id;
out.yTrue = cellfun(@(x) num2cell(x',1), tbl.yTrue);
out.yHat = cellfun(@(x) num2cell(x',1), tbl.yHat);


end