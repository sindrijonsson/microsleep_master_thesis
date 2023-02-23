clc
num = 0;
szX = [];
szY = [];
name = strings;
layer = strings;
for i = 1:numel(LSTM.Layers)-1
    l = LSTM.Layers(i);
    fn = fieldnames(l);
    w = fn(endsWith(fieldnames(l),"Weights") | endsWith(fn,"Bias"));

    if ~isempty(w)
        for j = 1:numel(w)
            fprintf("%s: %s\n",l.Name,w{j})
            name = [name; w{j}];
            layer = [layer; l.Name];
            x = size(l.(w{j}), 1);
            y = size(l.(w{j}), 2);
            szX = [szX; x]; szY = [szY; y];
            num = num + (x*y);
        end
    end
end
sz = [szX, szY];
name = name(2:end);
layer = layer(2:end);
t = table;
t.layer = layer;
t.name = name;
t.size = sz;
t.num = sz(:,1) .* sz(:,2);
t
fprintf("Total number of trainable parameters: %i\n", sum(t.num))