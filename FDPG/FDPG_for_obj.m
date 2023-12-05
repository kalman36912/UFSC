function [obj_val] = FDPG_for_obj(z, a, b, itr, reset)

seed = 12345;
rng(seed);

l = length(z);  
N = round((1 + sqrt(1+8*l))/ 2);
[S, St] = sum_squareform(N);

L = (N-1)/(b);
prox_g = @(x) (x+sqrt(x.^2+4*a*L))./2;

F.eval = @(x) (St*x)' * (max(0, (St*x - 2*z)./(2*b))) ...
        - 2 * (max(0, (St*x - 2*z)./(2*b)))' * z ...
        - b * norm((max(0, (St*x - 2*z)./(2*b))))^2;
G.eval = @(x) a * sum(log(a./x)) - a * N;

wk = rand(N,1);
yk = wk;
tk = 1;
for k=1:itr
    u = max(eps,(St*wk-2*z)./(2*b));
    v = prox_g((S*u-L*wk));
    yK = wk - (1/L).*(S*u-v);
    tK = (1+sqrt(1+4*(tk^2)))/2;
    wK = yK + ((tk-1)/tK)*(yK - yk);
    
    obj_val(k) = F.eval(yK) + G.eval(yK);
    
    tk = tK;
    yk = yK;
    wk = wK;
    if rem(k,reset) == 0
        tk = 1;
    end
end

end