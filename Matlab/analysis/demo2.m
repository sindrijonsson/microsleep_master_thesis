% for i = 42:numel(params.filePaths)
%     
%     view_rec(params.filePaths(i), 1);
%     pause
% 
% end

% C1Wu 500-530
% LR2s 890-900
% BKx9 1040-1050
% zaca 1360-1390

%%
view_rec("data/zaca.mat",1)
%%
xlim([1360, 1390])
xticklabels(0:5:30)
xlabel("Time [s]")
ch = findall(gcf,"Type","Axes");

for ci = 3:numel(ch)
    ax = ch(ci);
    if contains(ax.YLabel.String,"E")
        ax.YLim = [-100, 100];
        legend(ax,"off")
    end
    if contains(ax.YLabel.String,"O")
        ax.YLim = [-100, 100];
        legend(ax,"off")
    end
end

%% Remove MSC label
for ci = 3:4
    ax = ch(ci);
    for j = 1:numel(ax.Children)
        if ax.Children(j).Type == "patch"
            if ax.Children(j).FaceColor == [0,0,1]
                ax.Children(j).Visible = false;  
            end
        end
    end
end

%% Remove MS label
% for ci = 3:4
%     ax = ch(ci);
%     for j = 1:numel(ax.Children)
%         if ax.Children(j).Type == "patch"
%             if ax.Children(j).FaceColor == [0,1,0]
%                 ax.Children(j).Visible = false;  
%             end
%         end
%     end
% end

%%
cb = findall(gcf,"Type","ColorBar");
set(cb(1), "Position", [0.911284722082524,0.110072689506869,0.011111111111111,0.10280373962139])

%%
set(findall(gcf,"-property","FontSize"),"FontSize",16)
sgtitle("Microsleep example","FontSize",20)
    
%% 
d=load("data/zaca.mat");
data = d.Data;
nparams=params;
nparams.windowStepSec=0.2;
nparams.zeroPad = zeros(1,80);
f = generate_features(data, nparams);

%%
figure(2)
ix=(1:7)+(7*1);
lim = 1361:1391;
lim = lim*5;
fx = f(lim,ix);
% fx = f(l,ix);
ft = ["Delta","Theta","Alpha","Beta","T/AB","eye m.","med f."];
for i = 1:size(fx,2)
    subplot(size(fx,2),1,i)
    axis(gca,"tight")
    if i < 4; fx(:,i) = pow2db(fx(:,i)); end
    area(1360:1390, fx(:,i), "EdgeColor","none")
    ylabel(ft(i))
    xline(1376,'--k')
    xline(1385,'--k')
end