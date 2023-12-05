function W =updateW_FJGSED(X, U,mu,k)
data_num = size(X,1);
node_num = size(X,2);

xd= pdist(X').^2;
xd=xd(:);
xd = xd/data_num;
Xd = squareform(xd);

ud= pdist(U).^2;
ud=ud(:);
Ud = squareform(ud);

W = zeros(node_num,node_num);

for i = 1:node_num
    ci = Xd(i,:) + (mu/2)* Ud(i,:);
    [v,~] = sort(ci,'ascend');
    c_i_k_plus_1 = v(k+1);
    c_k_sum = sum(v(1:k));
    w_i = max((c_i_k_plus_1 - ci)./(k*c_i_k_plus_1 - c_k_sum),0);
    W(i,:) = w_i;
    
end

W = (W +W')/2;
W = W - diag(diag(W));
end