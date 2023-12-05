function [W, stat] = FDPG_log_degree(z, a, b, tol)

l = length(z);  
N = round((1 + sqrt(1+8*l))/ 2);
[S, St] = sum_squareform(N);

L = 1.0*((N-1)/b);
prox_g = @(x) (x+sqrt(x.^2+4*a*L))./2;

f.eval = @(w) b * norm(w)^2;
g.eval = @(w) 2 * w' * z - a * sum(log(S*w));

wk = randn(N,1);
temp = zeros(N*(N-1)/2,1);
temp2 = zeros(N,1);
yk = wk;
tk = 1;

tic
for k=1:5000
    u = max(eps,(St*wk-2*z)./(2*b));
    v = prox_g((S*u-L*wk));
    yK = wk - (1/L).*(S*u-v);
    tK = (1+sqrt(1+4*(tk^2)))/2;
    wK = yK + ((tk-1)/tK)*(yK - yk);
    
%     obj_val_ours(k) = g.eval(u) + f.eval(u);
    rel_norm_u = norm(u-temp,2)/norm(temp,2);
    rel_norm_w = norm(wK-temp2,2)/norm(temp2,2);
%     rel_norm_u = norm(u-temp,'fro');
%     rel_norm_w = norm(wK-temp2,'fro');
    if rel_norm_u<tol && rel_norm_w<tol
        break
    end
    
    tk = tK;
    yk = yK;
    wk = wK;
    
    
    temp = u;
    temp2 = wK;
end
stat.time = toc;
% stat.num_itr = length(obj_val_ours);
% stat.obj_val = obj_val_ours;
W = squareform_sp(u);
W(W<0.0001) = 0;
end