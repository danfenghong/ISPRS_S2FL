function P=Solving_P1(Y,Z1,alfa,maxiter)

%% 

%% Initializing Setting
[L,d]=size(Y);
[D,N]=size(Z1);
epsilon = 1e-6;
iter=0;

Q=zeros(L,D);
% Q1=zeros(size(Q));
% Q2=zeros(size(Y));

lamda1=zeros(size(Q));
% lamda2=zeros(size(Q));
% lamda3=zeros(size(Y));
% lamda4=zeros(size(theta*X));

stop = false;
mu=1e-3;
rho=1.5;
mu_bar=1e+6;

 while ~stop && iter < maxiter+1
    
      iter=iter+1;
      %solve P
%       P=(Y*Z1'+Y*Z2'+mu*Q+lamda1+mu*(Q1*Z1')+lamda2*Z1'+mu*(Q2*Z2')+lamda3*Z2')/(Z1*Z1'+Z2*Z2'+mu*eye(size(Z1*Z1'))+mu*(Z1*Z1')+mu*(Z2*Z2'));
        P=((Y*Z1')+mu*Q+lamda1)/((Z1*Z1')+mu*eye(size(Z1*Z1')));  
      %solve Q
       Q=max(abs(P-lamda1/mu)-(alfa/mu),0).*sign(P-lamda1/mu); 
%       r=P-lamda1/mu;
%       rL2=zeros(1,size(P,2));
%       for i=1:size(r,2)
%           rL2(1,i)=norm(r(:,i));
%           Q(:,i)=max(rL2(1,i)-alfa/mu,0)*(r(:,i)/rL2(1,i));    
%       end
%  Resi_M = P-lamda1/mu;
%     [U S V] = svd(Resi_M, 'econ');
%     diagS = diag(S);
%     svp = length(find(diagS > alfa/mu));
%     if svp>=1
%         diagS = diagS(1:svp)-alfa/mu;
%     else
%         svp = 1;
%         diagS = 0;
%     end
%     Q = U(:,1:svp)*diag(diagS)*V(:,1:svp)'; 

%       Q1=max(P*Z1-(lamda2/mu),0);
%       Q2=max(P*Z2-(lamda3/mu),0);
%       Q1=max(P-(lamda2/mu),0);  
%        T1=P-(lamda2/mu);
%            [U,S,V] = svd(T1);
%     Q1=U*eye(size(P))*V';
%        for i=1:size(T1,2)
%         if norm(T1(:,i))<=1
%             Q1(:,i)=T1(:,i);
%         else
%             Q1(:,i)=T1(:,i)/norm(T1(:,i));
%         end
%        end
%       
%        T2=P*Z2-(lamda3/mu);
%        for i=1:size(T2,2)
%         if norm(T2(:,i))<=1
%             Q2(:,i)=T2(:,i);
%         else
%             Q2(:,i)=T2(:,i)/norm(T2(:,i));
%         end
%        end
       
     %update Lagrange multipliers  
     lamda1=lamda1+mu*(Q-P);
%      lamda2=lamda2+mu*(Q1-P);
%      lamda3=lamda3+mu*(Q2-P*Z2);
%      lamda4=lamda4+beta*(P-theta*X);
     %update penalty parameter
     mu=min(mu*rho,mu_bar);
     %computer errors
     r_P=norm(Q-P,'fro');
%      r=norm(Y-P*Z1,'fro');
%      r_G=norm(Q1-P,'fro');
%      r_Q=norm(Q2-P*Z2,'fro');
%      r_P=norm(P-theta*X,'fro');
     %check the convergence conditions
     if r_P<epsilon%&&r<epsilon%&&r_G<epsilon%&&r_Q<epsilon%&&r_P<epsilon
         stop = true;
         break;
     end
  end
end