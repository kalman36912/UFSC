function W = FDPG_log_degree_denoising(signal, alpha, beta,xi, tol)

Xo = signal;
signal_num = size(Xo,1);
node_num  = size(Xo,2);
P = sparse(gsp_distanz(Xo).^2);
p = squareform_sp(P/signal_num);
threshold = 0.001;
W_last = rand(node_num);
for i = 1 : 100
    [W, ~] = FDPG_log_degree(p, alpha, beta, tol);
    L = diag(sum(W,2)) - W;
    X =(eye(node_num)+(1/xi) * L)\Xo';
    X = X';
    P = sparse(gsp_distanz(X).^2);
    p = squareform_sp(P/signal_num);
    
    error_w = norm(W(:)-W_last(:))/norm(W_last(:));
%     fprintf("error_w: %f\n: ", error_w);
    if (error_w<threshold)
        break
    else
        W_last = W;  
    end 
    
end








end