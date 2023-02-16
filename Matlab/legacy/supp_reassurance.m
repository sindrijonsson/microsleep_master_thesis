%% Reproduce plots from supplementary material
close all
clear all
clc
%% read splits
fid = fopen("..\\splits\\skorucack_splits.json"); 
raw = fread(fid,inf); 
str = char(raw'); 
fclose(fid); 
splits = jsondecode(str);


%%
list = string(ls("data"));
list = list(arrayfun(@(x) ~startsWith(x,"."), list));
names = replace(list,".mat","");
fullPaths = strcat("data\", list);

trainIdx = find(ismember(names, splits.train));
testIdx = find(ismember(names, splits.test));

%%
file = fullPaths(names=="csxQ")
tmp = load(file, "-mat");
data = tmp.Data;

y = [data.labels_O1, data.labels_O2]';
% y = apply_moving_func(y, 200, 0.2, 0.2, @(x) median(x,2));
time = linspace(0, 40, length(y));
% time_sec = seconds(time / data.fs);
% time_min = minutes(3600 * time / data.fs);

fmt = table;
fmt.label = ["W","ED","MSEc","MSEu","MSE"]';
fmt.val = [5, 4, 3, 2, 1]';
fmt.target = [0, 3, 2, 1, 1]';

figure(1);
clf
ax=gca;
for i = 1:height(fmt)
    row = fmt(i,:);
    if ismember(row.label,["W","MSE"])
        crit = @all;
    else
        crit = @any;
    end
    label = parse_bern(y, row.target, crit, row.val);
    if row.label=="MSEu"
        check = parse_bern(y, 1, @all, 1);
        label = label .* ~check;
    end
    
    label = apply_moving_func(label, 200, 0.2, 0.2, @(x) median(x,2));
    time = linspace(0, 40, length(label));
    [first, last] = get_first_and_last(label, row.val);
    plot_patches(first, last, row.val-1, row.val, time, ax)
end
ax.set("YTick",0.5:1:4.5);
ax.set("YTickLabels",flip(fmt.label));
ylim([0,5])
box on
fig =gcf;
fig.set("Position",[2125,365, 1598, 328])
xlabel("time (min)")

