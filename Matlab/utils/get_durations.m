function durs = get_durations(x, secPerSamp)

[first, last, singles] = get_first_and_last(x, 1);
durs = ((last - first)+1)*secPerSamp;
durs = [durs, ones(size(singles))*secPerSamp];


end