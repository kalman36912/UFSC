function [time] = DPG_for_time(z, a, b, w_star)

seed = 12345;
rng(seed);

l = length(z);  
N = round((1 + sqrt(1+8*l))/ 2);
[S, St] = sum_squareform(N);

L = (N-1)/(b);
prox_g = @(x) (x+sqrt(x.^2+4*a*L))./2;

f.eval = @(w) b * norm(w)^2;
g.eval = @(w) 2 * w' * z - a * sum(log(S*w));

yk = rand(N,1);
err_DPG = 1;

tic
while err_DPG > 1e-8
    u = max(eps,(St*yk-2*z)./(2*b));
    v = prox_g((S*u-L*yk));
    yK = yk - (1/L).*(S*u-v);
        
    yk = yK;
    W_hat = max(0, (St*yK - 2*z)./(2*b));
    err_DPG = norm(W_hat - w_star, 2);
end
time = toc;
end