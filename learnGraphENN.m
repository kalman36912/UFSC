function graph = learnGraphENN(X,rw,eps)
s = size(X);
N = s(1);
d = s(2);

r = pdist(X').^2/N;
s = exp(-r/(2*rw^2));
s(s>eps) = 0;

s(s~=0) = 1;

S = squareform(s);


graph = (S +S')/2;



end








