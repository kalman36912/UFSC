function graph = learnGraphKNN(X,rw,K)
s = size(X);
N = s(1);
d = s(2);

r = pdist(X').^2/N;
s = exp(-r/(2*rw^2));
S = squareform(s);

S = S + diag(1000*ones(d,1));

for i = 1:d
    row = S(i,:);
    [~,idx] = sort(row);
    ind_remove = idx(K+1:d);
    S(i,ind_remove) = 0;
end



graph = (S +S')/2;
graph(graph~=0) = 1;

end








