function U = updateU(UU,W,Q,R,mu,gamma)
U0 = UU;

opts.record = 0;
opts.mxitr  = 1000;
opts.xtol = 1e-5;
opts.gtol = 1e-5;
opts.ftol = 1e-8;
% out.tau = 1e-3;
%opts.nt = 1;

degrees = sum(W, 1);
D = diag(degrees);
L = D-W;

%profile on;
[U, ~]= OptStiefelGBB(U0, @funUpdateU, opts,L, Q,R, mu,gamma); 

function [F, G] = funUpdateU(U,L, Q,R, mu,gamma)
    G =  2 * mu *L*U - 2*gamma* Q*(R');
    F = mu * trace((U')*L*U)  - 2*gamma * trace(R * Q' * U);
end

end