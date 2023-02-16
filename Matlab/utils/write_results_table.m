function []=write_results_table(overallTable, perRecTable, type, params, prefix)

if nargin < 5; prefix = ""; else prefix=prefix+"_"; end

numCol = size(overallTable,2);

if strcmpi(type,"sample")
    %Write to table 
    fid = fopen(fullfile(params.outFolder,sprintf("%ssample.txt",prefix)),"w");
    f1 = "%s \t" + strjoin(repelem("%.2f",1,numCol-1)," \t ") + "\n";
    f2 = "%s \t" + strjoin(repelem("%s",1,numCol-1)," \t ")+ "\n";
    fprintf(fid,f2,string(overallTable.Properties.VariableNames));
    for i = 1:height(overallTable)
        fprintf(fid,f1,overallTable{i,:});
        fprintf(fid,f2,perRecTable{i,:});
    end
    fclose(fid);
end

if strcmpi(type,"event")
    %Write to table
    fid = fopen(fullfile(params.outFolder,sprintf("%sevent.txt",prefix)),"w");
    f1 = "%s \t" + strjoin(repelem("%.2f",1,numCol-1)," \t ") + "\n";
    f2 = "%s \t" + strjoin(repelem("%s",1,numCol-1)," \t ")+ "\n";
    fprintf(fid,f2,string(overallTable.Properties.VariableNames));
    for i = 1:height(overallTable)
        fprintf(fid,f1,overallTable{i,:});
        fprintf(fid,f2,perRecTable{i,:});
    end
    fclose(fid);
end

end