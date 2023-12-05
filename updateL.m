% Method of Dong2016
function graph = updateL(X, W, U,alpha,beta,mu)
data_num = size(X,1);
node_num = size(X,2);
w = squareform(W)';
w_i = w;

xd= pdist(X').^2;
xd=xd(:);
xd = xd/data_num;

ud= pdist(U).^2;
ud=ud(:);

z = xd + (mu/2)*ud;

num_iter = 2000;
S = genS(node_num);
w_last = w;
threshold = 0.0005;
eta = 0.0002;
for i = 1: num_iter

    grad = 2*z + 2*beta*w_i - alpha*S'* (1./(S*w_i+10e-6));  
    w_i = w_i - eta * grad;
    w_i(w_i<0) = 0;
    if norm(w_i-w_last)/norm(w_last)<threshold
        break
    else
        w_last = w_i;  
    end    
end


w_i(w_i<0.001) = 0;
graph = squareform(w_i);


end

