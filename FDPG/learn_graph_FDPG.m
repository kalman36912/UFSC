function [W_hat, stat] = learn_graph_FDPG(Z, a, b, params)

%=========================================================================%
% Usage information                                                       %
%=========================================================================%
% This function learns a graph from pairwise distances using negative log %
% prior on nodes degrees; for more information see [Kalofolias, Vassilis. %
% "How to learn a graph from smooth signals." Artificial Intelligence and %
% Statistics. PMLR, 2016.]                                                %
%   Usage:  [W, stat] = learn_graph_FDPG(Z, a, b)                         %
%           [W, stat] = learn_graph_FDPG(Z, a, b, params)                 %
%                                                                         %
%   Inputs:                                                               %
%           Z       : Matrix (or vector of upper triangular elements)     %
%                     with (squared) pairwise distances of nodes          %
%           a       : Log prior constant (bigger a -> bigger weights in W)%
%           b       : ||W||_F^2 prior constant (bigger b -> more dense W) %
%           params  : Optional parameters                                 %
%                                                                         %
%   Outputs:                                                              %
%           W       : Thresholded weighted adjacency matrix               %
%           stat    : Optional output statistics                          %
%                                                                         %
%   Optional parameters:                                                  %
%       params.verbos   : Default = 0. 1 will print more information      %
%       params.maxitr   : Default = 2000. Maximum number of iterations    %
%       params.tol      : Default = 1e-6. Tolerance for stopping criterion%
%       params.thr      : Default = 1e-5. Threshold for removing weak edge%
%       params.reset    : Default = 100. Restart interval after a specific%
%                         number of iterations; see [Odonoghue, Brendan, %
%                         and Emmanuel Candes. "Adaptive restart for      %
%                         accelerated gradient schemes." Foundations of   %
%                         computational mathematics 15.3 (2015): 715-732.]% 
%                                                                         %
%       Note: To run this function properly you need to install the GSPbox%
%             which is a MATLAB toolbox and can be found at               %              
%             "https://epfl-lts2.github.io/gspbox-html/"                  %
%                                                                         %
%       References:                                                       %
%           Seyed Saman Saboksayr and Gonzalo Mateos,                     %
%           "Accelerated graph learning from smooth signals."             %
%           IEEE Signal Processing Letters (2021).  
%           Preprint: https://arxiv.org/abs/2110.09677
%                                                                         %
%   *** If you have used this function please kindly cite our paper ***   %
%                                                                         %
%       Author: Seyed Saman Saboksayr                                     %
%       Date: August 2021                                                 %
%=========================================================================%

seed = 12345;
rng(seed);

%=========================================================================%
% Default parameters                                                      %
%=========================================================================%

if nargin < 4
    params = struct;
end

if not(isfield(params, 'verbos')),          params.verbos = 0;          end
if not(isfield(params, 'maxitr')),          params.maxitr = 2000;       end
if not(isfield(params, 'tol')),             params.tol = 1e-6;          end
if not(isfield(params, 'reset')),           params.reset = 100;         end
if not(isfield(params, 'thr')),             params.thr = 1e-5;          end

%=========================================================================%
% Fix parameters                                                          %
%=========================================================================%

if isvector(Z)
    z = Z;  
else
    z = squareform_sp(Z);
end
z = z(:);
l = length(z);  
N = round((1 + sqrt(1+8*l))/ 2);

%=========================================================================%
% Required functions                                                      %
%=========================================================================%

[S, St] = sum_squareform(N);
L = (N-1)/(b);

prox_g = @(x) (x+sqrt(x.^2+4*a*L))./2;
f.eval = @(w) b * norm(w)^2 + 2 * w' * z;
g.eval = @(w) - a * sum(log(S*w));

F.eval = @(x) (St*x)' * (max(0, (St*x - 2*z)./(2*b))) ...
        - 2 * (max(0, (St*x - 2*z)./(2*b)))' * z ...
        - b * norm((max(0, (St*x - 2*z)./(2*b))))^2;
G.eval = @(x) a * sum(log(a./x)) - a * N;


%=========================================================================%
% FDPG algorithm                                                          %
%=========================================================================%

if nargout > 1
    stat.obj_val      = nan(params.maxitr, 1);
    stat.dual_obj_val = nan(params.maxitr, 1);
end

omega_k = rand(N,1);
lambda_k = omega_k;
w_hat_k = max(eps, (St*lambda_k - 2*z)./(2*b));
tk = 1;

tic
for k = 1:params.maxitr
    
    w_bar  = max(eps,(St*omega_k-2*z)./(2*b));
    u  = prox_g((S*w_bar-L*omega_k));
    lambda_K = omega_k - (1/L).*(S*w_bar-u);
    tK = (1+sqrt(1+4*(tk^2)))/2;
    omega_K = lambda_K + ((tk-1)/tK)*(lambda_K - lambda_k);
    w_hat_K = max(eps, (St*lambda_K - 2*z)./(2*b));
    
    rel_norm_w   = norm(w_hat_K - w_hat_k, 'fro')/norm(w_hat_k, 'fro');
    rel_norm_lam = norm(lambda_K - lambda_k, 'fro')/norm(lambda_k);
        
    tk = tK;
    lambda_k = lambda_K;
    omega_k = omega_K;
    w_hat_k = w_hat_K;
    
    if nargout > 1 || params.verbos == 1
    stat.obj_val(k)      = g.eval(w_hat_K) + f.eval(w_hat_K);
    stat.dual_obj_val(k) = F.eval(lambda_K) + G.eval(lambda_K);
    end
    if params.verbos == 1
        fprintf('iteration %4d: %6.4e *** %6.4e *** %6.3e \n', k, rel_norm_w, rel_norm_lam, stat.obj_val(k));
    end
    
    if rem(k,params.reset) == 0
        tk = 1;
    end
    
    if rel_norm_w < params.tol && rel_norm_lam < params.tol
        break
    end
end
stat.time   = toc;
stat.Lambda = lambda_K;
if params.verbos == 1
    fprintf('Done! \n');
end
stat.obj_val(find(isnan(stat.obj_val)))           = [];
stat.dual_obj_val(find(isnan(stat.dual_obj_val))) = [];
%=========================================================================%
% Thresholding the weighted learned graph                                 %
%=========================================================================%

w_hat_K(w_hat_K < params.thr) = 0;
W_hat = squareform_sp(w_hat_K);

%=========================================================================%
end