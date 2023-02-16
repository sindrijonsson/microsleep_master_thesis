function probs = get_probs(file)
load(file,"data");
probs = data ./ sum(data,2);

end