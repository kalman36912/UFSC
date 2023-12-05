% Method of Dong2016
function graph = updateL_FDPG(X, W, U,alpha,beta,mu)
signal_num = size(X,1);


Xd = sparse(gsp_distanz(X).^2);
xd = squareform_sp(Xd/signal_num);

Ud = sparse(gsp_distanz(U').^2);
ud = squareform_sp(Ud);

p = xd + (mu/2)*ud;

tol = 0.0001;
[W, stat] = FDPG_log_degree(p, alpha, beta, tol);

graph = W;

end

