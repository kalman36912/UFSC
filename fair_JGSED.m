function [W,res_lables, obj] =fair_JGSED(X,F, K, mu, gamma, node_neighbour)


% alpha,beta, gamma : parameters
threshold = 0.001;
data_num = size(X,1);
node_num = size(X,2);

%%Initialize
xd= pdist(X').^2;
xd=xd(:);
xd = xd/data_num;
Xd = squareform(xd);

W = zeros(node_num,node_num);
for i = 1 : node_num
    ci = Xd(i,:);
    [v,~] = sort(ci,'ascend');
    c_i_k_plus_1 = v(node_neighbour+1);
    c_k_sum = sum(v(1:node_neighbour));
    w_i = max((c_i_k_plus_1 - ci)./(node_neighbour*c_i_k_plus_1 - c_k_sum),0);
    W(i,:) = w_i;
end
W = (W +W')/2;
% L = diag(diag(W)) - W ;


U = Fair_SC_unnormalized(W,K,F);
Z = null(F');
Y = (Z')*U;
lables_init = kmeans(U,K,'Replicates',10);
Q = zeros(node_num, K);
for i = 1:length(lables_init)
    Q(i,lables_init(i)) = 1;
end
R = updateR(Q,U);


node_neighbour = 10;

% xd= pdist(X').^2;
% xd=xd(:);
% xd = xd/data_num;

num_iter = 100;

obj = zeros(num_iter,1);

W_last = W;


for i = 1: num_iter
%     fprintf("iter: %d\n: ", i);


    %% Update L/W
    W = updateW_FJGSED(X, U,mu,node_neighbour);
%     L = diag(sum(W,2)) - W;
    %% Update U/Y
    Y = updateY(Y,Z,W,Q,R,mu,gamma);
    U = Z*Y;
    %% Update R
    R = updateR(Q,U);
    %% Update Q
    Q = updateQ(U,R);
    
    error_w = norm(W(:)-W_last(:))/norm(W_last(:));
%     fprintf("error_w: %f\n: ", error_w);
    if (error_w<threshold)
        break
    else
        W_last = W;  
    end  
    
%     obj(i) = 1/data_num * (trace(X*L*X')) - alpha * sum(log(diag(L))) + beta * norm(W,'fro')^2 + mu*trace(U'*L*U) + gamma * norm(Q-U*R,'fro')^2;
end
res_lables = zeros(node_num,1);
for i = 1 :node_num
    res_lables(i) = find(Q(i,:)~=0);
    
end


end