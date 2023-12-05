function S = genS(N)
S = zeros(N,N*(N-1)/2);
for i = 1: N-1
    for j = (i-1)*(2*N-i)/2+1:(i-1)*(2*N-i)/2+(N-i)
        S(i,j) = 1; 
    end    
end
for i = 1:N-1
    k = i+1;
    for j = (i-1)*(2*N-i)/2+1:(i-1)*(2*N-i)/2+(N-i)
        S(k,j)=1;
        k = k+1;
    end
end
end