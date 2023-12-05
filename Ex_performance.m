%% Test file
clear all 
clc


alpha  = 1;


a=0.8;
b=0.2;
c=0.15;
d=0.05;


cluster_num = 4; %% #cluster
sensitve_attr_num = 2; %% #sensitivity attribute

num_metrics = 4;
num_experiments = 1;
num_baslines = 6; %% Other baselines are implemented by python in open  package
node_num =  192; %# graph nodes
signal_num = 5000;
noise_level = 0.5;
block_sizes = (node_num/(cluster_num*sensitve_attr_num))*ones(1,cluster_num*sensitve_attr_num);


%generate graphs 
W_real = generate_adja_SB_model(node_num,a,b,c,d,cluster_num,sensitve_attr_num,block_sizes);
%generate graph signals
signal = generate_signals(W_real,signal_num,noise_level);

%Initial
xd= pdist(signal').^2;
xd=xd(:);
xd = xd/signal_num;
W = rand(node_num);
W = (W + W')/2;
W=W-diag(diag(W));
W_FSRSC = W;
W = FDPG_log_degree(xd, alpha, 5, 0.0001);

sensitive=zeros(node_num,1);
labels=zeros(node_num,1);


for yyy=1:cluster_num
    for zzz=1:sensitve_attr_num
        sensitive(((node_num/cluster_num)*(yyy-1)+(node_num/(cluster_num*sensitve_attr_num))*(zzz-1)+1):((node_num/cluster_num)*(yyy-1)+(node_num/(cluster_num*sensitve_attr_num))*zzz))=zzz;
        labels(((node_num/cluster_num)*(yyy-1)+(node_num/(cluster_num*sensitve_attr_num))*(zzz-1)+1):((node_num/cluster_num)*(yyy-1)+(node_num/(cluster_num*sensitve_attr_num))*zzz))=yyy;
    end
end


%% converting sensitive to a vector with entries in [h] and building F %%%
sens_unique=unique(sensitive);
h = length(sens_unique);
sens_unique=reshape(sens_unique,[1,h]);
sensitiveNEW=sensitive;
temp=1;
for ell=sens_unique
    sensitiveNEW(sensitive==ell)=temp;
    temp=temp+1;
end  
F=zeros(node_num,h-1);
for ell=1:(h-1)
    temp=(sensitiveNEW == ell);
    F(temp,ell)=1; 
    groupSize = sum(temp);
    F(:,ell) = F(:,ell)-groupSize/node_num;
end
Z = null(F');

%%%%
res_all = zeros(num_baslines,num_metrics, num_experiments);
for k = 1: num_experiments


%% Our methods, noise are homogenous. If the noise are heterogenous, use fair_GL_SC_heter
fprintf("Ours\n");

beta = 3;
xi = 5;
mu = 0.5;
gamma = 0.01;

[W_ours,res_labels_ours,~] = fair_GL_SC(signal, F, W,cluster_num, alpha,beta,mu, gamma,xi);
W_ours = W_ours/sum(sum(W_ours))*node_num;



%% FJGSED
fprintf("FJGSED\n");
node_neighbour = 50;
mu_FJGSED =  0.005 ;
gamma_FJGSED = 0.0005;
[W_FJGSED,res_labels_FJGSED,~] = fair_JGSED(signal, F, cluster_num, mu_FJGSED, gamma_FJGSED,node_neighbour);
W_FJGSED = W_FJGSED/sum(sum(W_FJGSED))*node_num;


%% FSRSC
fprintf("FSRSC\n");
alpha_FSRSC = 0.16;
mu_FSRSC = 0.0001;
gamma_FSRSC = 0.00005;

[W_FSRSC,res_labels_FSRSC] = fair_SRSC(signal, F, W_FSRSC,cluster_num, alpha_FSRSC, mu_FSRSC, gamma_FSRSC);
W_FSRSC = W_FSRSC/sum(sum(W_FSRSC))*node_num;





%% Kmeans
fprintf("Kmeans\n");
res_labels_kmeans = kmeans(signal',cluster_num,'Replicates',10);



%% kNN
fprintf("kNN\n");
K_NN = 20;
rw = 1;
W_KNN = learnGraphKNN(signal,rw,K_NN);
H_KNN =  Fair_SC_unnormalized(W_KNN,cluster_num,F);
res_labels_KNN = kmeans(H_KNN,cluster_num,'Replicates',10);
W_KNN = W_KNN/sum(sum(W_KNN))*node_num;


%% ENN
fprintf("EpsNN\n");
eps = 0.3;
rw = 1;
W_ENN = learnGraphENN(signal,rw,eps);
H_ENN =  Fair_SC_unnormalized(W_ENN,cluster_num,F);
res_labels_ENN = kmeans(H_ENN,cluster_num,'Replicates',10);
W_ENN = W_ENN/sum(sum(W_ENN))*node_num;


topo_ours = evaluator (1, W_real, W_ours, node_num);
topo_FSRSC = evaluator (1, W_real, W_FSRSC, node_num);
topo_FJGSED = evaluator (1, W_real, W_FJGSED, node_num);
topo_KNN = evaluator (1, W_real, W_KNN, node_num);
topo_ENN = evaluator (1, W_real, W_ENN, node_num);



fs_ours = topo_ours(3);
fs_FSRSC = topo_FSRSC(3);
fs_FJGSED = topo_FJGSED(3);
fs_ENN = topo_ENN(3);
fs_KNN = topo_KNN(3);





re_ours = evaluator (2, W_real, W_ours,Z);
re_FSRSC = evaluator (2, W_real, W_FSRSC,Z);
re_FJGSED = evaluator (2, W_real, W_FJGSED,Z);
% re_kmeans = evaluator (2, W_real, W_kmeans);
re_KNN = evaluator (2, W_real, W_KNN,Z);
re_ENN = evaluator (2, W_real, W_ENN,Z);



cluster_error_ours = evaluator (3, labels, res_labels_ours);
cluster_error_FSRSC = evaluator (3, labels, res_labels_FSRSC);
cluster_error_FJGSED = evaluator (3, labels, res_labels_FJGSED);
cluster_error_kmeans = evaluator (3, labels, res_labels_kmeans);
cluster_error_KNN = evaluator (3, labels, res_labels_KNN);
cluster_error_ENN = evaluator (3, labels, res_labels_ENN);


balance_ours = evaluator (4, sensitive, res_labels_ours);
balance_FSRSC = evaluator (4, sensitive, res_labels_FSRSC);
balance_FJGSED = evaluator (4, sensitive, res_labels_FJGSED);
balance_kmeans = evaluator (4, sensitive, res_labels_kmeans);
balance_KNN = evaluator (4, sensitive, res_labels_KNN);
balance_ENN = evaluator (4, sensitive, res_labels_ENN);


res_kmeans = [-1,-1, cluster_error_kmeans, balance_kmeans];

res_FSRSC = [fs_FSRSC, re_FSRSC, cluster_error_FSRSC, balance_FSRSC];
res_FJGSED = [fs_FJGSED,re_FJGSED, cluster_error_FJGSED, balance_FJGSED];


res_KNN = [fs_KNN,re_KNN, cluster_error_KNN, balance_KNN];
res_ENN = [fs_ENN, re_ENN, cluster_error_ENN, balance_ENN];

res_ours = [fs_ours,re_ours, cluster_error_ours, balance_ours];

res_one_ex = [res_kmeans;   res_KNN; res_ENN;   res_FJGSED; res_FSRSC; res_ours];


res_all(:,:, k) = res_one_ex;

end

res_mean = squeeze(mean(res_all,3));












