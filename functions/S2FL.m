function [theta00, theta12, P, res] = S2FL(Y, X, Z, alfa, beta, L, maxiter)

epsilon = 1e-4; % Tolerance error
iter=1; 
stop = false;
res=zeros(1,maxiter); % Residuals
theta00 = zeros(size(Z,1), size(X,1));

while ~stop && iter < maxiter+1
    
    %% Solve P
        P = (Y * Z')/(Z * Z' + alfa * eye(size(Z * Z'))); 
        
    %% Solve the group of theta    
        theta12 = Solving_Theta12(Y - P * theta00 * X, P, X, maxiter);
        theta00 = Solving_Theta00(Y - P * theta12 * X, P, X, beta, maxiter, L);
        Z=(theta00 + theta12) * X;
    
    %% Compute loss   
     res(1,iter) = 0.5*norm(Y-P*(theta00 + theta12)*X,'fro')^2 + 0.5*alfa*norm(P,'fro')^2 + 0.5*beta*trace(theta00*X*L*X'*theta00');
   
    %% Check the convergence condition
    if iter>1
       r_Obj=abs(res(1,iter)-res(1,iter-1))/abs(res(1,iter-1));
       if r_Obj<epsilon
            stop = true;
            fprintf(' i = %f,res_Obj= %f\n',iter,r_Obj);
            break;
       end

       if mod(iter,10) == 1
           fprintf(' i = %f,res_Obj= %f\n',iter,r_Obj);
       end
    end

    iter=iter+1;
end
end