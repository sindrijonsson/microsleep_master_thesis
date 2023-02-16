view_rec("data/C1Wu.mat",1);

%%
ch = findall(gcf,"Type","Axes");
%%
ax = ch(1);
ax.XLabel.String = "Time [s]";
xlim([500, 530])

%%
for ci = 3:numel(ch)
    ax = ch(ci);
    if contains(ax.YLabel.String,"E")
        ax.YLim = [-100, 100];
        ax.YLabel.String = strcat(ax.YLabel.String, "(\muV)");
        legend(ax,"off")
    end
    if contains(ax.YLabel.String,"O")
        ax.YLim = [-100, 100];
        ax.YLabel.String = strcat(ax.YLabel.String, "(\muV)");
        legend(ax,"off")
    end
end

%%

% for ci = 3:4
%     ax = ch(ci);
%     for j = 1:numel(ax.Children)
%         if ax.Children(j).Type == "patch"
%             ax.Children(j).Visible = false;
%         end
%     end
% end

%%
set(findall(gcf,"-property","FontSize"),"FontSize",9)