function [obj_val] = DPG_for_obj(z, a, b, itr)

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

yk = rand(N,1);
for k=1:itr
    u = max(eps,(St*yk-2*z)./(2*b));
    v = prox_g((S*u-L*yk));
    yK = yk - (1/L).*(S*u-v);
    
    obj_val(k) = F.eval(yK) + G.eval(yK);
    
    yk = yK;
end

end