function signal =generate_signals_heter(A,signal_num,noise_level)

node_num = size(A,1);
deg_vec = sum(A,2);
laplacian = diag(deg_vec) - A;
mu = zeros(node_num,1);
sigma_noise = diag(noise_level.^2);
sigma = pinv(laplacian)+sigma_noise;
X = mvnrnd(mu,sigma,signal_num);
% noise=zeros(signal_num, node_num);
% for i =1 :node_num
%     noise(:,i) = normrnd(0,noise_level(i),signal_num,1);
% end
% X = X + noise;
signal = X;



end