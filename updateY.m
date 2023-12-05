function Y = updateY(YY,Z,W,Q,R,mu,gamma)
Y0 = YY;

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
[Y, ~]= OptStiefelGBB(Y0, @funUpdateY, opts,L, Z,Q,R, mu,gamma); 

function [F, G] = funUpdateY(Y, L, Z,Q,R, mu,gamma)
%     G = 2*gamma*(Z')*Z*Y*R*(R') -2 *gamma*(Z')*Q*(R') + 2 * mu *Z'*L *Z *Y;
%     F = mu * trace((Y')*(Z')*L*Z*Y) - 2*gamma * trace(R * Q' * Z *Y);
    G =  2 * mu *Z'*L *Z *Y - 2 *gamma*(Z')*Q*(R') ;
    F = mu * trace((Y')*(Z')*L*Z*Y) - 2*gamma * trace(R * Q' * Z *Y);
end

end