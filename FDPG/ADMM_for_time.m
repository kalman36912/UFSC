function [time] = ADMM_for_time(z, a, b, step1, step2, step3, w_star)
seed = 12345;
rng(seed);

l = length(z);  
N = round((1 + sqrt(1+8*l))/ 2);
[S, St] = sum_squareform(N);
sigma = svd(full(S));

t = step1;
tu1 = step2/(sigma(1));
tu2 = step3;

wk = rand(N*(N-1)/2,1);
vk = rand(N,1);
lamk = rand(N,1);
err_ADMM = 1;

tic
while err_ADMM > 1e-8
    wtild = (wk - tu1.*St*(S*wk - vk - lamk./t) - 2*tu1*z)./(2*tu1*b + 1);
    wK = max(0, wtild);
    vtild = (1-tu2*t).*vk + tu2*t*(S*wK) - tu2.*lamk./t;
    vK = 0.5.*(vtild + sqrt(vtild.^2 + 4*a*tu2));
    lamK = lamk - t*(S*wK - vK);
    
    wk = wK;
    vk = vK;
    lamk = lamK;
    err_ADMM = norm(wK - w_star, 2);
end
time = toc;
end