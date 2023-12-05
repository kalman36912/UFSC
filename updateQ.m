function Q  = updateQ(U,R)
T = U*R;
dim = size(T);
Q = zeros(dim);
for i = 1 : dim(1)
    tmp = T(i,:);
    [~,index] = max(tmp);
    Q(i,index) = 1;
end


end