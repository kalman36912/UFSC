function signal =generate_signals(A,signal_num,noise_level)

node_num = size(A,1);
deg_vec = sum(A,2);
laplacian = diag(deg_vec) - A;
mu = zeros(node_num,1);
sigma_noise = (noise_level^2)*eye(node_num);
sigma = pinv(laplacian)+sigma_noise;
X = mvnrnd(mu,sigma,signal_num);
signal = X;



end