function [w, stat, err_kal] = kal_for_error(Z, a, b, max_itr, step, w_star)
seed = 12345;
rng(seed);
%% Default parameters
params = struct;

if not(isfield(params, 'verbosity')),   params.verbosity = 0;   end
if not(isfield(params, 'maxit')),       params.maxit = max_itr;      end
if not(isfield(params, 'tol')),         params.tol = 1e-40;      end
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

if isfield(params, 'w_0')
    if not(isfield(params, 'c'))
        error('When params.w_0 is specified, params.c should also be specified');
    else
        c = params.c;
    end
    if isvector(params.w_0)
        w_0 = params.w_0;
    else
        w_0 = squareform_sp(params.w_0);
    end
    w_0 = w_0(:);
else
    w_0 = 0;
end

% if sparsity pattern is fixed we optimize with respect to a smaller number
% of variables, all included in w
if params.fix_zeros
    if not(isvector(params.edge_mask))
        params.edge_mask = squareform_sp(params.edge_mask);
    end
    % use only the non-zero elements to optimize
    ind = find(params.edge_mask(:));
    z = full(z(ind));
    if not(isscalar(w_0))
        w_0 = full(w_0(ind));
    end
else
    z = full(z);
    w_0 = full(w_0);
end


w = rand(size(z));

%% Needed operators
% S*w = sum(W)
if params.fix_zeros
    [S, St] = sum_squareform(n, params.edge_mask(:));
else
    [S, St] = sum_squareform(n);
end

% S: edges -> nodes
K_op = @(w) S*w;

% S': nodes -> edges
Kt_op = @(z) St*z;

if params.fix_zeros
    norm_K = normest(S);
    % approximation: 
    % sqrt(2*(n-1)) * sqrt(nnz(params.edge_mask) / (n*(n+1)/2)) /sqrt(2)
else
    % the next is an upper bound if params.fix_zeros
    norm_K = sqrt(2*(n-1));
end

%% TODO: Rescaling??
% we want    h.beta == norm_K   (see definition of mu)
% we can multiply all terms by s = norm_K/2*b so new h.beta==2*b*s==norm_K


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

if w_0 == 0
    % "if" not needed, for c = 0 both are the same but gains speed
    h.eval = @(w) b * norm(w)^2;
    h.grad = @(w) 2 * b * w;
    h.beta = 2 * b;
else
    h.eval = @(w) b * norm(w)^2 + c * norm(w - w_0,'fro')^2;
    h.grad = @(w) 2 * ((b+c) * w - c * w_0);
    h.beta = 2 * (b+c);
end

%% My custom FBF based primal dual (see [1] = [Komodakis, Pesquet])
% parameters mu, epsilon for convergence (see [1])
mu = h.beta + norm_K;     %TODO: is it squared or not??
epsilon = lin_map(0.0, [0, 1/(1+mu)], [0,1]);   % in (0, 1/(1+mu) )

% INITIALIZATION
% primal variable ALREADY INITIALIZED
%w = params.w_init;
% dual variable
v_n = K_op(w);
if nargout > 1 || params.verbosity > 1
    stat.f_eval = nan(params.maxit, 1);
    stat.g_eval = nan(params.maxit, 1);
    stat.h_eval = nan(params.maxit, 1);
    stat.fgh_eval = nan(params.maxit, 1);
    stat.pos_violation = nan(params.maxit, 1);
end
if params.verbosity > 1
    fprintf('Relative change of primal, dual variables, and objective fun\n');
end

tic
gn = lin_map(params.step_size, [epsilon, (1-epsilon)/mu], [0,1]);              % in [epsilon, (1-epsilon)/mu]
for i = 1:max_itr
    Y_n = w - gn * (h.grad(w) + Kt_op(v_n));
    y_n = v_n + gn * (K_op(w));
    P_n = f.prox(Y_n, gn);
    p_n = g_star_prox(y_n, gn); % = y_n - gn*g_prox(y_n/gn, 1/gn)
    Q_n = P_n - gn * (h.grad(P_n) + Kt_op(p_n));
    q_n = p_n + gn * (K_op(P_n));
    
    if nargout > 1 || params.verbosity > 2
        stat.f_eval(i) = f.eval(w);
        stat.g_eval(i) = g.eval(K_op(w));
        stat.h_eval(i) = h.eval(w);
        stat.fgh_eval(i) = stat.f_eval(i) + stat.g_eval(i) + stat.h_eval(i);
        stat.pos_violation(i) = -sum(min(0,w));
    end
    
    w = w - Y_n + Q_n;
    v_n = v_n - y_n + q_n;
    
    err_kal(i) = norm(w - w_star, 2);
end
stat.time = toc;
if params.verbosity > 0
    fprintf('# iters: %4d. Rel primal: %6.4e Rel dual: %6.4e  OBJ %6.3e\n', i, rel_norm_primal, rel_norm_dual, f.eval(w) + g.eval(K_op(w)) + h.eval(w));
    fprintf('Time needed is %f seconds\n', stat.time);
end


%%

if params.verbosity > 3
    figure; plot(real([stat.f_eval, stat.g_eval, stat.h_eval])); hold all; plot(real(stat.fgh_eval), '.'); legend('f', 'g', 'h', 'f+g+h');
    figure; plot(stat.pos_violation); title('sum of negative (invalid) values per iteration')
    figure; semilogy(max(0,-diff(real(stat.fgh_eval'))),'b.-'); hold on; semilogy(max(0,diff(real(stat.fgh_eval'))),'ro-'); title('|f(i)-f(i-1)|'); legend('going down','going up');
end

end