function [time] = kal_for_time(Z, a, b, step, w_star)
seed = 12345;
rng(seed);
%% Default parameters
params = struct;
if not(isfield(params, 'verbosity')),   params.verbosity = 0;   end
if not(isfield(params, 'fix_zeros')),   params.fix_zeros = false;      end
if not(isfield(params, 'max_w')),       params.max_w = inf;         end
params.step_size = step;
%% Fix parameter size and initialize
if isvector(Z)
    z = Z;  % lazy copying of matlab doesn't allocate new memory for z
else
    z = squareform_sp(Z);
end
% clear Z   % for large scale computation

z = z(:);
l = length(z);                      % number of edges
% n(n-1)/2 = l => n = (1 + sqrt(1+8*l))/ 2
n = round((1 + sqrt(1+8*l))/ 2);    % number of nodes
w = rand(size(z));

%% Needed operators
% S*w = sum(W)

[S, St] = sum_squareform(n);


% S: edges -> nodes
K_op = @(w) S*w;

% S': nodes -> edges
Kt_op = @(z) St*z;
norm_K = sqrt(2*(n-1));

%% Learn the graph
% min_{W>=0}     tr(X'*L*X) - gc * sum(log(sum(W))) + gp * norm(W-W0,'fro')^2, where L = diag(sum(W))-W
% min_W       I{W>=0} + W(:)'*Dx(:)  - gc * sum(log(sum(W))) + gp * norm(W-W0,'fro')^2
% min_W                f(W)          +       g(L_op(W))      +   h(W)

% put proximal of trace plus positivity together
f.eval = @(w) 2*w'*z;    % half should be counted
%f.eval = @(W) 0;
f.prox = @(w, c) min(params.max_w, max(0, w - 2*c*z));  % all change the same

param_prox_log.verbose = params.verbosity - 3;
g.eval = @(z) -a * sum(log(z));
g.prox = @(z, c) prox_sum_log(z, c*a, param_prox_log);
% proximal of conjugate of g: z-c*g.prox(z/c, 1/c)
g_star_prox = @(z, c) z - c*a * prox_sum_log(z/(c*a), 1/(c*a), param_prox_log);

h.eval = @(w) b * norm(w)^2;
h.grad = @(w) 2 * b * w;
h.beta = 2 * b;

%% My custom FBF based primal dual (see [1] = [Komodakis, Pesquet])
% parameters mu, epsilon for convergence (see [1])
mu = h.beta + norm_K;     %TODO: is it squared or not??
epsilon = lin_map(0.0, [0, 1/(1+mu)], [0,1]);   % in (0, 1/(1+mu) )

% INITIALIZATION
% primal variable ALREADY INITIALIZED
%w = params.w_init;
% dual variable
v_n = K_op(w);
err_kal = 1;

tic
gn = lin_map(params.step_size, [epsilon, (1-epsilon)/mu], [0,1]);              % in [epsilon, (1-epsilon)/mu]
while err_kal > 1e-8
    Y_n = w - gn * (h.grad(w) + Kt_op(v_n));
    y_n = v_n + gn * (K_op(w));
    P_n = f.prox(Y_n, gn);
    p_n = g_star_prox(y_n, gn); % = y_n - gn*g_prox(y_n/gn, 1/gn)
    Q_n = P_n - gn * (h.grad(P_n) + Kt_op(p_n));
    q_n = p_n + gn * (K_op(P_n));
    
    w = w - Y_n + Q_n;
    v_n = v_n - y_n + q_n;
    
    err_kal = norm(w - w_star, 2);
end
time = toc;
end