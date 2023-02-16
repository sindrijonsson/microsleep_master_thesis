

function stats = get_target_hit_duration_table(inp, params)
dTable=table;
for i=1:height(inp)
    [a,b,c] = get_target_hit_durations(inp{i,"yTrue"},inp{i,"yHat"},params.secPerLabel);
    if size(a,2) > size(a,1); a=a'; b=b'; c=c'; end
    tmpTable = table;
    tmpTable.dur = a;
    tmpTable.hit = b;
    tmpTable.total = c;
    if ~isempty(a)
        dTable = [dTable; tmpTable];
    end
end

stats = rowfun(@(hit, total) sum(hit)/sum(total), dTable, ...
    "InputVariables",["hit","total"], ...
    "GroupingVariables","dur", ...
    "OutputVariableNames","perc");
end
