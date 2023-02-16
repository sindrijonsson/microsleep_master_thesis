
p = cell2mat(testMuSleep.probs);
pTest = postTable(ismember(postTable.id,params.split.test),:);
pTest.c = (1:height(pTest))';
mt = table;
mt.p = p(:,2);
mt.id = pTest.id;
mt.c = pTest.c;
pTest.MSEu(isnan(pTest.MSEu)) = 0;
cls = ["W","ED","MSEc","MSEu","MSE"];
t = join(mt, pTest, "Keys", ["id","c"], "RightVariables",cls);
%%
tt = stack(t,cls,"IndexVariableName","class","NewDataVariableName","isClass");
T=tt(find(tt.isClass), :);
T.class = categorical(T.class);
boxplot(T.p, T.class);

%%
scatter(T,"class","p","Marker",".","XJitter","density")

%%
err="std";

gs=grpstats(T,"class",["mean",err],"DataVars","p");
errorbar(gs.class, gs.mean_p, gs.(err+"_p"), "o", "MarkerFaceColor", colors.BLUE, "MarkerSize", 12);

%%
clc
a=unstack(T, "p", "class");
a = removevars(a,["id", "isClass","c"]);
v=violinplot(a);


