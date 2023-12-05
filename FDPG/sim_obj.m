close all; clearvars; clc;

seed = 0; % choosing seed for reproducibility of figures
rng(seed); % for different results you may change the seed

max_itr = 100; % number of iterations
N = 200; % number of nodes
clust_nodes = [100,100]; % number of nodes in each block
P = 0.05*ones(2) + diag(0.25*ones(2,1)); % connection probability matrix

A1 = SBM_graph_gen(N,clust_nodes,P); % generating SBM graph
nSamples = 1000; % number of i.i.d. graph signal samples
L = diag(sum(A1,2)) - A1;

[V,D] = eig(L);
d = pinv(D);

sigma = 0.1;
mu = zeros(1,N);
gftcoeff = mvnrnd(mu,d,nSamples);
X = V*gftcoeff';
X_noisy = X + sigma*randn(size(X));

Z = sparse(gsp_distanz(X_noisy').^2);
z = squareform_sp(Z/nSamples);

%==========================================================================
% grid search for finding alpha and beta
%==========================================================================

a = 10.^[-1:1:1];
b = 10.^[-1:1:1];

idx = 1;
for i=1:length(a)
    for j=1:length(b)
        param(:,idx) = [a(i);b(j)];
        idx = idx + 1;
    end
end

for t=1:size(param,2)
    [w_test,~] = learn_graph_FDPG(Z/nSamples, param(1,t), param(2,t));
    [~,~,F(t),~,~] = graph_learning_perf_eval(A1,w_test);
end
idx = find(F==max(F));
idx = idx(end);
a = param(1,idx);
b = param(2,idx);

params.maxitr = 50000;
params.tol = 1e-40;
params.reset = 10;
[~, stat] = learn_graph_FDPG(Z/nSamples, a, b, params);
phi_star  = stat.dual_obj_val(end);

%==========================================================================
% FDPG
%==========================================================================

reset = 10;
[obj_FDPG] = FDPG_for_obj(z, a, b, max_itr, reset);
phi_FPDG = abs(obj_FDPG - phi_star);


%==========================================================================
% DPG
%==========================================================================

[obj_DPG] = DPG_for_obj(z, a, b, max_itr);
phi_PDG = abs(obj_DPG - phi_star);

%==========================================================================
% Convergence figure
%==========================================================================

h1 = figure(1);
hold on
p1 = plot(phi_FPDG, '-', 'LineWidth',1,'Color','#026440');
hold on
p2 = plot(phi_PDG, '-.', 'LineWidth',1,'Color','#D79922');

grid on
ax = gca;
ax.GridLineStyle = '--';
ax.GridColor = '#ADADAD';
ax.GridAlpha = 0.75;
legend([p1 p2], {'FDPG', 'DPG'},'Location','northeast')
xlabel('Number of iterations')
ylabel('$ \varphi (\mathbf{\lambda}_k) - \varphi (\mathbf{\lambda}^{\star}) $','Interpreter','Latex')
set(gca, 'YScale','log')