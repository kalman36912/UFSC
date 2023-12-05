close all; clearvars; clc;

seed = 0; % choosing seed for reproducibility of figures
rng(seed); % for different results you may change the seed

load('brain_data_66.mat','CC'); % loading the data

nSamples = 1000; % number of i.i.d. graph signal samples
patient = 6; % patient id

G = reshape(CC(:,:,patient),[66,66]); % Scale by 100
thr = 1e-4;
A1 = double((G>=thr)); % thresholding
N = size(G,1); % number of nodes

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

a = 10.^[-1:0.25:2];
b = 10.^[-1:0.25:2];

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

reset = 100;
time_FDPG = FDPG_for_time(z, a, b, reset, w_star);

%==========================================================================
% DPG
%==========================================================================

time_DPG = DPG_for_time(z, a, b, w_star);

%==========================================================================
% PD
%==========================================================================

t1 = 0.05:0.1:0.95;
for i=1:size(t1,2)
    time_PD(i) = kal_for_time(z, a, b, t1(i), w_star);
end
time_PD = min(time_PD);

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
    [W_ADMM, stat_ADMM] = ADMM_for_error(z, a, b, 2000, step(1,i), step(2,i), step(3,i));
    temp4_err(i) = norm(W_ADMM{2000} - w_star, 2);
end

idx = find(temp4_err==min(temp4_err));
idx = idx(1);
time_ADMM = ADMM_for_time(z, a, b, step(1,idx), step(2,idx), step(3,idx), w_star);

%==========================================================================
% Printing results
%==========================================================================

fprintf('Wall-clock time for FDPG: %7f \n', time_FDPG);
fprintf('Wall-clock time for DPG: %7f \n', time_DPG);
fprintf('Wall-clock time for PD: %7f \n', time_PD);
fprintf('Wall-clock time for ADMM: %7f \n', time_ADMM);