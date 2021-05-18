function theta=Solving_Theta(Y,P,H,X,beta,maxiter,L)
%solve / optimize the variable theta
%H is defined as theta * X

epsilon = 1e-6;
iter=0;

[l,d]=size(H);
[D,N]=size(X);

G=zeros(l,D);
theta=zeros(size(G));
lamda1=zeros(size(H));
lamda2=zeros(size(G));

stop = false;
mu=1E-3;
rho=1.5;
mu_bar=1E+6;

GL=X*L*X';

while ~stop && iter < maxiter+1
    
    iter=iter+1;
    
    %update H
    H = (P'*P+mu*eye(size(P'*P)))\(P'*Y+mu*(theta*X)-lamda1);
    
    %update theta
    theta = (mu*(H*X')+lamda1*X'+mu*G+lamda2)/(mu*(X*X')+mu*eye(size(X*X'))+beta*GL);  
    
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
