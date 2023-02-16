function k = calc_kappa(x)
    m=size(x,1);
    f=diag(ones(1,m)); %unweighted
    n=sum(x(:)); %Sum of Matrix elements
    x=x./n; %proportion
    r=sum(x,2); %rows sum
    s=sum(x); %columns sum
    Ex=r*s; %expected proportion for random agree
    po=sum(sum(x.*f)); %proportion observed
    pe=sum(sum(Ex.*f)); %proportion expected
    k=(po-pe)/(1-pe); %Cohen's kappa
end