close all; clearvars; clc;

seed = 0; % choosing seed for reproducibility of figures
rng(seed); % for different results you may change the seed

load('minnesota.mat'); % loading the data

A1 = full(double(Problem.A>0));
N = size(A1,1);

max_itr = 5000; % number of iterations
nSamples = 5000; % number of i.i.d. graph signal samples
L = diag(sum(A1,2)) - A1;

[V,D] = eig(L);
d = pinv(D);

thr = 1e-5;

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
    partest.maxit = 5000; 
    [w_test,~] = gsp_learn_graph_log_degrees(Z/nSamples, param(1,t), param(2,t),partest);
    w_test(w_test<thr) = 0;
    [~,~,F(t),~,~] = graph_learning_perf_eval(A1,w_test);
end
idx = find(F==max(F));
idx = idx(end);
a = param(1,idx);
b = param(2,idx);

params.maxit = 50000; 
params.tol = 1e-40;
params.step_size = 0.5;
[w_star, stat_star] = gsp_learn_graph_log_degrees(z, a, b, params);
w_star(w_star<thr) = 0;

%==========================================================================
% FDPG
%==========================================================================

reset = 550;
[W_FDPG, stat_FDPG, err_FDPG] = FDPG_for_error(z, a, b, max_itr, reset, w_star);

%==========================================================================
% DPG
%==========================================================================

[W_DPG, stat_DPG, err_DPG] = DPG_for_error(z, a, b, max_itr, w_star);

%==========================================================================
% PD
%==========================================================================

[W_kal, stat_kal, err_kal] = kal_for_error(z, a, b, max_itr, 0.95, w_star);

%==========================================================================
% Convergence figure
%==========================================================================


h1 = figure(1);
hold on
p1 = plot(err_FDPG, '-', 'LineWidth',1,'Color','#026440');
hold on
p2 = plot(err_kal, '--', 'LineWidth',1,'Color','#1A1A1D');
hold on
p3 = plot(err_DPG, '-.', 'LineWidth',1,'Color','#D79922');

grid on
ax = gca;
ax.GridLineStyle = '--';
ax.GridColor = '#ADADAD';
ax.GridAlpha = 0.75;
legend([p1 p2 p3], {'FDPG','PD','DPG'},'Location','northeast')
xlabel('Number of iterations')
ylabel('$\| \hat{w}_{k} - w^{\star}\|_{2}$','Interpreter','Latex')
set(gca, 'YScale','log')