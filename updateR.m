function R = updateR(Q,U)
T = (Q')*U;
[A,~,B] = svd(T);
R = A * (B');
end