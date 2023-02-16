folder = fullfile("benchmark_test_correct_freq_res");
files = string(ls(fullfile(folder,"*.txt")));
files = strip(fullfile(folder, files));

allTable = table;

for i = 1:numel(files)
    f = files(i);
    opts = detectImportOptions(f);
    opts = setvartype(opts, opts.VariableNames, "string");
    tbl = readtable(f, opts);
    comparison = f.extractBefore(".txt").extractAfter("\");
    tbl.comparison = repelem(comparison, height(tbl), 1);
    allTable = [allTable; tbl];
end
allTable = renamevars(allTable,"recall","sensitivity");
allTable = allTable(:,["sensitivity","specificity","precision","accuracy","kappa","comparison"]);
writetable(allTable, "all_tables.xlsx","WriteMode","overwrite");