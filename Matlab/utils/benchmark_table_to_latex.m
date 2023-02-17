

function benchmark_table_to_latex(file)

opts = detectImportOptions(file);
opts = setvartype(opts, opts.VariableNames, 'string');  %or 'char' if you prefer
data = readtable(file, opts);

vars = ["recall", "specificity","precision","accuracy","kappa"];

outFile = file.replace(".txt","_latex.txt");

fid = fopen(outFile,"w");

first = @(x) fprintf(fid, "\\multicolumn{1}{l}{%s} &\n",x);
second = @(x) fprintf(fid, "(%s) &\n",x.replace("+/-","$\pm$"));
last = @(x) fprintf(fid, "(%s) \\\\ \\hline \n",x.replace("+/-","$\pm$"));

for i = 1:numel(vars)
    v = vars(i);
    for j = 1:2:height(data)
        if j == 1
           fprintf(fid,"\\textbf{This work} & %% %s\n",v)
        end
        f = data{j,v};
        first(f)
        
        s = data{j+1,v};
        if (j + 1) == height(data)
            last(s);
        else
            second(s);
        end
    end
    fprintf(fid,"\n")
end
fclose(fid)
end