function [W, stat] = PG_for_error(z, a, b, itr)

seed = 12345;
rng(seed);

l = length(z);  
N = round((1 + sqrt(1+8*l))/ 2);
[S, St] = sum_squareform(N);
w = rand(N*(N-1)/2,1);

f.eval = @(w) 2*w'*z;
f.prox = @(w,c) max(eps, w - 2.*c.*z);
g.eval = @(w) b * norm(w)^2 -a * sum(log(S*w));
g.grad = @(w) 2 * b * w - a * St * (1./(S*w));
g.beta = @(w) 2 * b + (2*a*(N-1))/(min(S*w)^2);

tic
for i=1:itr
    step = 2/g.beta(w);
    w_k = f.prox(w - step * g.grad(w),step);
    obj_val(i) = g.eval(w_k) + f.eval(w_k);
    
    w = w_k;
    W{i} = w_k;
end
stat.time = toc;
stat.num_itr = length(obj_val);
stat.obj_val = obj_val;
end