t = uSleep_perRecMetrics;

tbl = stack(t, t.Properties.VariableNames(1:end-1), ...
    "IndexVariableName", "metric", ...
    "NewDataVariableName","value");
tbl = tbl(tbl.metric ~= "fpMinutes",:);
tbl.metric = ordinal(tbl.metric);
tbl.id = categorical(tbl.id);
s=scatter(tbl, "metric","value","filled")
s.ColorVariable = "id";
colorbar