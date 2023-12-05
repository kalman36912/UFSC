clear all 
clc
alpha  = 1;
beta =2;
gamma = 1;

a=0.4;
b=0.3;
c=0.2;
d=0.1;
cluster_num = 4; %% #cluster
sensitve_attr_num = 2; %% #sensitivity attribute


node_num_list = [96, 192, 288, 384, 480, 576, 672, 768, 864, 960];
node_num =  960; %# graph nodes

signal_num = 35000;
noise_level = 0;

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
% W = learnGraph(xd,alpha,beta, W);

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
%%%%



%%

% learn graphs 
tic
Z = sparse(gsp_distanz(signal).^2);
z = squareform_sp(Z/signal_num);
toc
tol = 0.0001;




alpha_list = [1];
beta_list = [0.1, 0.5,0.8,1,2,4,6,8,10,20,50,100];
num_alpha = length(alpha_list);
num_beta = length(beta_list);
fs_list = zeros(num_alpha, num_beta);


best_fs = 0;
best_alpha =0;
best_beta = 0;
for i = 1: num_alpha
    for j = 1: num_beta
        alpha = alpha_list(i);
        beta = beta_list(j);
        tic
        [W, stat] = FDPG_log_degree(z, alpha, beta, tol);
        toc
        topo_ours = evaluator (1, W_real, W, node_num);
        fs_ours = topo_ours(3);
        if best_fs<fs_ours
            best_fs = fs_ours;
            best_alpha = alpha;
            best_beta = beta;
            
        end
        
        fs_list(i,j) = fs_ours;
        
    end
    
end


plot(beta_list, fs_list)
set(gca,'XScale','log','FontSize',18)
% figure;
% [xx,yy]=meshgrid(alpha_list, beta_list); 
% 
% surf(xx,yy,fs_list)
% shading interp;
% xl = xlabel('$\alpha$','fontsize',18);
% set(xl,'Interpreter','latex')
% yl = ylabel('$\beta$','fontsize',18);
% set(yl,'Interpreter','latex')
% zlabel('\fontname{Times New Roman}FS','fontsize',18);
% ax = gca;
% set(gca,'XScale','log','YScale','log','FontSize',18)
% 
% % ylim([lambda_list(1),lambda_list(num_lambda_list)]);
% % xlim([rho_list(1),rho_list(num_rho_list)]);
% 
% 
% 
