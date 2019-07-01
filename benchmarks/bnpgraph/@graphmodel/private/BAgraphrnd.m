function G = BAgraphrnd(n)

%BAgraphrnd samples a graph from the Barabasi-Albert (BA) model
% G = BAgraphrnd(n)


G = zeros(n,n);

for nn=1:n
    p = zeros(1,nn-1);
    for kk=1:nn-1
        p(kk) = sum(G(kk,:))/2;
    end
    if sum(p)==0
        p(1) = 1;
    else
        p = p./sum(p);
    end
    for kk=1:nn-1
        G(nn,kk) = (rand(1)<p(kk));
        G(kk,nn) = G(nn,kk);
    end
end
