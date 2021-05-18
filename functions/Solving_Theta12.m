function theta=Solving_Theta12(Y,P,X,maxiter)
%solve / optimize the variable theta

epsilon = 1e-6;
iter=0;

[~,l]=size(P);
[D,N]=size(X);

G=zeros(l,D);
theta=zeros(size(G));
lamda1=zeros(l,N);
lamda2=zeros(size(G));

stop = false;
mu=1E-3;
rho=1.5;
mu_bar=1E+6;

while ~stop && iter < maxiter+1
    
    iter=iter+1;
    
    %update H
    H = (P'*P+mu*eye(size(P'*P)))\(P'*Y+mu*(theta*X)-lamda1);
    
    %update theta
    theta = (mu*(H*X')+lamda1*X'+mu*G+lamda2)/(mu*(X*X')+mu*eye(size(X*X')));  
    %theta = (mu*(H*X')+lamda1*X')/(mu*(X*X'));
    
    %update G
    Q = theta-lamda2/mu;  
    [U,~,V] = svd(Q);
    G = U*eye(size(theta))*V';
    
    %update Lagrange multipliers
    lamda1 = lamda1+mu*(H-theta*X);
    lamda2 = lamda2+mu*(G-theta);
    
    %update penalty parameter
    mu = min(mu*rho,mu_bar);
  
    %compute errors
    r_H = norm(H-theta*X,'fro');
    r_G = norm(G-theta,'fro');
    
    %check the convergence conditions
    if r_H<epsilon&&r_G<epsilon%&&r_M<epsilon%&&r_Q<epsilon%
        stop = true;
        break;
    end
    
end
end
