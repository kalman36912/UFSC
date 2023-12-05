close all; clearvars; clc;

seed = 0; % choosing seed for reproducibility of figures
rng(seed); % for different results you may change the seed

max_itr = 1000; % number of iterations
N = 200; % number of nodes
[A1,~, ~] = construct_graph(N,'er',0.1); % generating ER graph with p = 0.1
A1 = full(A1);
nSamples = 1000; % number of i.i.d. graph signal samples
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

a = 10.^[-1:0.2:2];
b = 10.^[-1:0.2:2];

idx = 1;
for i=1:length(a)
    for j=1:length(b)
        param(:,idx) = [a(i);b(j)];
        idx = idx + 1;
    end
end

for t=1:size(param,2)
    [w_test,~] = gsp_learn_graph_log_degrees(Z/nSamples, param(1,t), param(2,t));
    w_test(w_test<thr) = 0;
    [~,~,F(t),~,~] = graph_learning_perf_eval(A1,w_test);
end
idx = find(F==max(F));
idx = idx(end);
a = param(1,idx);
b = param(2,idx);

params.maxit = 50000; 
params.tol = 1e-40;
params.step_size = 0.1;
[w_star, ~] = gsp_learn_graph_log_degrees(z, a, b, params);

%==========================================================================
% FDPG
%==========================================================================

reset = 50;
[W_FDPG, stat_FDPG, err_FDPG] = FDPG_for_error(z, a, b, max_itr, reset, w_star);

%==========================================================================
% DPG
%==========================================================================

[W_DPG, stat_DPG, err_DPG] = DPG_for_error(z, a, b, max_itr, w_star);

%==========================================================================
% PG
%==========================================================================

[W_PG, stat_PG] = PG_for_error(z, a, b, max_itr);

for i=1:max_itr
    err_PG(i) = norm(W_PG{i} - w_star, 2);
end

%==========================================================================
% PD
%==========================================================================
t1 = 0.05:0.1:0.95;

for i=1:size(t1,2)
    [~, ~, temp] = kal_for_error(z, a, b, max_itr, t1(i), w_star);
    temp2_err(i) = temp(end);
end

idx = find(temp2_err==min(temp2_err));
idx = idx(1);
[W_kal, stat_kal, err_kal] = kal_for_error(z, a, b, max_itr, t1(idx), w_star);

%==========================================================================
% ADMM
%==========================================================================
t1 = 10.^[-1:0.25:2];
t2 = 0.05:0.1:0.95;
t3 = 0.05:0.1:0.95;
idx = 1;
for i=1:length(t1)
    for j=1:length(t2)
        for k=1:length(t3)
            step(:,idx) = [t1(i);t2(j);t3(k)];
            idx = idx + 1;
        end
    end
end

parfor i=1:size(step,2)
    [W_ADMM, stat_ADMM] = ADMM_for_error(z, a, b, max_itr, step(1,i), step(2,i), step(3,i));
    temp4_err(i) = norm(W_ADMM{max_itr} - w_star, 2);
end

idx = find(temp4_err==min(temp4_err));
idx = idx(1);
[W_ADMM, stat_ADMM] = ADMM_for_error(z, a, b, max_itr, step(1,idx), step(2,idx), step(3,idx));

for i=1:max_itr
    err_ADMM(i) = norm(W_ADMM{i} - w_star, 2);
end


%==========================================================================
% Convergence figure
%==========================================================================

h1 = figure(1);
hold on
p1 = plot(err_PG, '-.', 'LineWidth',1.0,'Color','#C3073F');
hold on
p2 = plot(err_FDPG, '-', 'LineWidth',1.0,'Color','#026440');
hold on
p3 = plot(err_kal, '--', 'LineWidth',1.0,'Color','#1A1A1D');
hold on
p4 = plot(err_ADMM, '-', 'LineWidth',1.0,'Color','#000080');
hold on
p5 = plot(err_DPG, '-.', 'LineWidth',1.0,'Color','#D79922');

grid on
ax = gca;
ax.GridLineStyle = '--';
ax.GridColor = '#ADADAD';
ax.GridAlpha = 0.75;
legend([p1 p2 p3 p4 p5], {'PG','FDPG','PD','ADMM', 'DPG'},'Location','northeast')
xlabel('Number of iterations')
ylabel('$\| \hat{w}_{k} - w^{\star}\|_{2}$','Interpreter','Latex')
set(gca, 'YScale','log')