clc
x = -10:0.01:10;

relu = @(x) max(0,x);
sigmoid = @(x) 1./(1+exp(-x));
myTanh = @(x) tanh(x);

fn = ["Sigmoid", "ReLU", "tanh", "ELU"];
figure(1); clf; hold on
for i = 1:numel(fcns)
    name = fn(i);
    switch i
        case 1
            y = sigmoid(x);
        case 2
            y = relu(x);
        case 3
            y = myTanh(x);
        case 4
            y = elu(x);
    end
    subplot(1,4,i)
    plot(x, y, "LineWidth", 2.5, "DisplayName", name);
    title(name)
    grid on
    if i == 4; 
        ylim([-2 10]);
    elseif i == 2;
        ax = gca;
        yl = ax.YLim;
        
        ylim([yl(1)-0.5, yl(2)])
    else
        ax = gca;
        yl = ax.YLim;
        s=0.1;
        ylim([yl(1)-s, yl(2)+s])
    end
    xlim([min(x), max(x)])
end

set(findall(gcf,"-property","FontSize"),"FontSize",12)

function out = elu(x)
% if nargin < 2; alpha = 1.0; end
out = zeros(size(x,1), size(x,2));
for i = 1:numel(x)
    if x(i) >= 0
        out(i) = x(i);
    else
        out(i) = 1.0*(exp(x(i)) - 1);
    end
end
end