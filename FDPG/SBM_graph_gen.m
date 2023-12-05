function [A] = SBM_graph_gen(N,clust_nodes,P)

A = tril(rand(N), -1);
C = cumsum(clust_nodes);
C = [1 C];

for i = 1:size(P,1)
    for j=1:i
        prob = P(i,j);
        row1 = C(i);
        row2 = C(i+1);
        col1 = C(j);
        col2 = C(j+1);
        
        A(row1:row2,col1:col2) = (A(row1:row2,col1:col2)<=prob);
    end
end

A = tril(A, -1);
A = A + A';