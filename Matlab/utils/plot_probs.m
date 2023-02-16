function plot_probs(probs, time, ax)

if nargin < 3
    figure(1);
    ax = gca;
end

% AASM
AASM = ["Wake", "N1", "N2", "N3", "REM"];

if height(probs) < length(time)
    probs = [probs;
             nan(length(time)-height(probs),size(probs,2))];
end

area(ax, time, probs(:,2:end),"FaceAlpha",1,"EdgeColor","none")
p = 0:0.5:1;
yticks(ax,p)
yticklabels(ax,string(p))
legend(AASM(2:end), "NumColumns",4, "Location","northwest","AutoUpdate","off")

end