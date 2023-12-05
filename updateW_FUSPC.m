% Method of Dong2016
function [W,G] = updateW_FUSPC(X, W, U,G, alpha,mu)
data_num = size(X,1);
node_num = size(X,2);

xi = 0.2;



J = W - G/xi;
%update S
S = max(abs(J) - alpha/xi,0) .* sign(J);
S = S -diag(diag(S));
S = max(S,0);


%update W
J_tilde = S + G/xi;
XTX = X'*X;
I = eye(node_num);
W = zeros(node_num, node_num);
P = pinv(2*XTX + xi*I);

ud= pdist(U).^2;
ud=ud(:);
Ud = squareform(ud);

for i = 1: node_num
    
    J_tilde_i = J_tilde(:,i);
    XTXi = XTX(i,:);
    Udi = Ud(:,i);
    W(:,i) = P * ( xi * J_tilde_i + 2*XTXi' -(mu/2)*Udi);
   
end

W = W - diag(diag(W));
W(W<0) = 0;
G = G + xi*(S-W);


W = (W +W')/2;

end

