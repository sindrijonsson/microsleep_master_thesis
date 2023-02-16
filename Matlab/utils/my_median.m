function out =  my_median(x)

    out = median(x,2);
    out(out==0.5) = 1;

end