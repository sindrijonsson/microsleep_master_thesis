files = string(ls(".\probs\"));
files = files(files.endsWith(".mat"));
files = strcat(".\probs\",files)

%%
prob_names = {"E1-M1 + O1-M2", "E1-M1 + O2-M1", "E2-M1 + O1-M2", "E2-M1 + O2-M1"};

fi = 17;
for fi = 1:numel(files)
    load(files(fi),"probs");
    
    view_rec(files(fi).replace(".\probs\","data\"),0);
    f=gcf;
    figure(2); clf;
    ax=subplot(6,1,1);
    n=files(fi).extractAfter("probs\").extractBefore(".mat");
    tmp = recLabels{rec==n};
    tmp = get_labels(files(fi).replace(".\probs\","data\"));
    time = linspace(0,2400,40*60*200);
    make_skorucack_label_plots(tmp, time, ax);
    box on
    for i = 1:size(probs,1)
        ax=subplot(size(probs,1)+2,1,i+1);
        tmp = squeeze(probs(i,:,:));
        time = linspace(0, 2400, 2400);
        plot_probs(tmp, time, ax)
        ylabel(ax, prob_names(i));
        ylim(ax, [0, 1])
        legend off
    end
    ax=subplot(6,1,6);
    plot_probs(squeeze(mean(probs,1)), time, ax); hold on;
    ylim(ax, [0, 1])
    yline(ax, optThres, '--k');
    linkaxes([findall(gcf,"Type","Axes"); findall(f,"Type","Axes")],"x")
    sgtitle(n)
    pause
end

%%
for fi = 1:numel(files)
    load(files(fi),"prbobs");
    probs = prbobs;

    figure(1); clf;
    ax=subplot(6,1,1);
    n=files(fi).extractAfter("probs\").extractBefore(".mat");
    tmp = recLabels{rec==n};
    time = linspace(0,40,40*60);
    plot_predictions(tmp, time, colors.BLUE, 1, ax);
    box on
    for i = 1:size(probs,1)
        ax=subplot(size(probs,1)+2,1,i+1);
        tmp = squeeze(probs(i,:,:));
        time = linspace(0, 40, 40*60);
        plot_probs(tmp, time, ax)
        ylabel(prob_names(i), "Rotation", 90);
        legend off
    end
    ax=subplot(6,1,6);
    plot_probs(squeeze(mean(probs,1)), time, ax); hold on;
    yline(ax, optThres);
    linkaxes(findall(gcf,"Type","Axes"))
    sgtitle(n)
    pause
end
%%
s=[];
for fi = 1:numel(files)
    load(files(fi),"prbobs");
    probs = prbobs;
    a = arrayfun(@(x) corr(squeeze(probs(:,:,x))'), 1:5, 'UniformOutput', false);
    s=[s; cell2struct(a,cellstr(AASM),2)];
end
