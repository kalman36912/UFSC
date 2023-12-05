function [time] = FDPG_for_time(z, a, b, reset, w_star)

seed = 12345;
rng(seed);

l = length(z);  
N = round((1 + sqrt(1+8*l))/ 2);
[S, St] = sum_squareform(N);

L = (N-1)/(b);
prox_g = @(x) (x+sqrt(x.^2+4*a*L))./2;

f.eval = @(w) b * norm(w)^2;
g.eval = @(w) 2 * w' * z - a * sum(log(S*w));

wk = rand(N,1);
yk = wk;
tk = 1;
err_FDPG = 1;
k = 1;

tic
while err_FDPG > 1e-8
    u = max(eps,(St*wk-2*z)./(2*b));
    v = prox_g((S*u-L*wk));
    yK = wk - (1/L).*(S*u-v);
    tK = (1+sqrt(1+4*(tk^2)))/2;
    
    wK = yK + ((tk-1)/tK)*(yK - yk);
        
    tk = tK;
    yk = yK;
    wk = wK;
    
    W_hat = max(0, (St*yK - 2*z)./(2*b));
    err_FDPG = norm(W_hat - w_star, 2);
    
    if rem(k,reset) == 0
        tk = 1;
    end
    k = k + 1;
end
time = toc;
end