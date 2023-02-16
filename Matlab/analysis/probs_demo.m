probs = ls("predictions\1_hz\*.mat");
probs = fullfile("predictions/1_hz/", string(probs));
probs = probs(contains(probs,["DjrT.mat","oOMR.mat"]))


for i = 1:numel(probs)
    figure(i); clf;
    file = load(probs(i));
    t = linspace(0,40,40*60);
    p = file.data ./ sum(file.data,2);
    plot_probs(p, t, gca)
%     title(probs(i));
    ylim([0,1])
    yticks(0:0.2:1)
    yticklabels(0:0.2:1)
    xlabel("Time [min]")
%     ylabel("P(S)")
%     xticks([]); xticklabels([]);
%     yticks([]); yticklabels([]);
end