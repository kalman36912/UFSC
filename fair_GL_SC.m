function [W,res_lables, obj] =fair_GL_SC(Xo,F, W, K, alpha,beta, mu, gamma,xi)
% X:data  N * d
% s: sensitive attribute   d * 1
% W : adjacency matrix    d * d
% H : cluster index matrix    d * k
% alpha,beta, gamma : parameters
threshold = 0.001;
data_num = size(Xo,1);
node_num = size(Xo,2);

%%Initialize
U = Fair_SC_unnormalized(W,K,F);
Z = null(F');
Y = (Z')*U;
lables_init = kmeans(U,K,'Replicates',10);
Q = zeros(node_num, K);
for i = 1:length(lables_init)
    Q(i,lables_init(i)) = 1;
end
R = updateR(Q,U);


num_iter = 40;

obj = zeros(num_iter,1);

W_last = W;
U_last = U;
X = Xo;

for i = 1: num_iter
%     fprintf("iter: %d\n: ", i);


    %% Update L/W
    W = updateL_FDPG(X, W, U,alpha,beta,mu);
    L = diag(sum(W,2)) - W;
    %% Update U/Y
    Y = updateY(Y,Z,W,Q,R,mu,gamma);
    U = Z*Y;
    %% Update R
    R = updateR(Q,U);
    %% Update Q
    Q = updateQ(U,R);
    
    X =(eye(node_num)+(1/xi) * L)\Xo';
    X = X';
    
    error_w = norm(W(:)-W_last(:))/norm(W_last(:));
    error_u = norm(U(:)-U_last(:))/norm(U_last(:));
%     fprintf("error_w: %f\n: ", error_w);
    if (error_w<threshold)
        break
    else
        W_last = W;  
        U_last = U;
    end  
    
  obj(i) = xi* (1/data_num) * norm(X(:) - Xo(:)) + 1/data_num * (trace(X*L*X')) - alpha * sum(log(diag(L))) + beta * norm(W,'fro')^2 + mu*trace(U'*L*U) + gamma * norm(Q-U*R,'fro')^2;
end
res_lables = zeros(node_num,1);
for i = 1 :node_num
    res_lables(i) = find(Q(i,:)~=0);
    
end


end