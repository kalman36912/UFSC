function H = Fair_SC_unnormalized(adj,k,F)
%
%INPUT:
%adj ... (weighted) adjacency matrix of size n x n
%k ... number of clusters
%sensitive ... vector of length n encoding the sensitive attribute 
%
%OUTPUT: U

n = size(adj, 1);

degrees = sum(adj, 1);
D = diag(degrees);
L = D-adj;

Z = null(F');

Msymm=Z'*L*Z;
Msymm=(Msymm+Msymm')/2;

try
    [Y, eigValues] = eigs(Msymm,k,'smallestabs','MaxIterations',500,'SubspaceDimension',min(size(Msymm,1),max(2*k,25)));
catch
    [Y, eigValues] = eigs(Msymm,k,'smallestreal','MaxIterations',1000,'SubspaceDimension',min(size(Msymm,1),max(2*k,25)));
end

H = Z*Y;

% clusterLabels = kmeans(H,k,'Replicates',10);


end