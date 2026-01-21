%DIEGO ALVAREZ FLORES
%BY REPLICATION: ONE STATE VARIABLE
%VFI: TAUCHEN
%JANUARY 2026

clear;
clc;
seed=202601;

%% PARAMETERS
sigma=0.0078;
mu=0.0015;
rho=0.979;
psi=1.5;
gamma=10;
beta=0.998;
phi_e=0.044;
theta=(1-gamma)/(1-1/psi);
phi=3;
mu_d=0.0015;
phi_d=4.5;

%% EQUATIONS FOR FIXED POINT METHOD (C0NSUMPTION)
function [k0] = function1_k0(zbar,k1)
    k0=log(exp(zbar)+1)-k1*zbar;
end

function [k1] = function1_k1(zbar)
    k1=exp(zbar)/(exp(zbar)+1);
end

function [A1] = function1_A1(psi,k1,rho)
    A1=(1-1/psi)/(1-k1*rho);
end

function [A0] = function1_A0(beta,mu,psi,k0,k1,theta,sigma,A1,phi_e)
    A0=(log(beta)+mu*(1-1/psi)+k0+0.5*sigma^2*theta*((1-1/psi)^2+(k1*A1*phi_e)^2))/(1-k1);
end

%% FIXED-POINT METHOD (C0NSUMPTION)
function [k0,k1,A0,A1,zbar] = solve_param_cons1(beta,mu,psi,theta,rho,phi_e,sigma)
    zbar_init=0;
    max_iter=1000;
    terminate=0;
    crit=1e-9; 
    iter=1;
    while (terminate==0 & iter<max_iter)
        k1 = function1_k1(zbar_init);
        k0 = function1_k0(zbar_init,k1);
        A1 = function1_A1(psi,k1,rho);
        A0 = function1_A0(beta,mu,psi,k0,k1,theta,sigma,A1,phi_e);
        zbar = A0; % Unconditional expectation of z_t
        if abs(zbar-zbar_init)<crit
            terminate=1;
        end        
        zbar_init=zbar;
        iter=iter+1;
    end
end

[k0,k1,A0,A1,zbar] = solve_param_cons1(beta,mu,psi,theta,rho,phi_e,sigma);

%% EQUATIONS FOR FIXED POINT METHOD (DIVIDEND)
function [k0m] = function1_k0m(zbarm,k1m)
    k0m=log(exp(zbarm)+1)-k1m*zbarm;
end

function [k1m] = function1_k1m(zbarm)
    k1m=exp(zbarm)/(exp(zbarm)+1);
end

function [A1m] = function1_A1m(psi,k1m,rho,phi)
    A1m=(phi-1/psi)/(1-k1m*rho);
end

function [A0m] = function1_A0m(beta,mu,psi,k0,k1,theta,sigma,A0,A1,k0m,k1m,A1m,mu_d,phi_e,phi_d)
    A0m = (theta*log(beta)+mu*(theta-1-theta/psi)+(theta-1)*(k0+A0*(k1-1)) ...
        +k0m+mu_d+0.5*sigma^2*((theta-1-theta/psi)^2+((theta-1)*k1*A1*phi_e+k1m*A1m*phi_e)^2+phi_d^2))/(1-k1m);
end

%% FIXED-POINT METHOD (DIVIDEND)
function [k0m,k1m,A0m,A1m,zbarm] = solve_param_div1(beta,mu,psi,theta,rho,phi_e,sigma, ...
        phi,k0,k1,A0,A1,phi_d,mu_d)
    zbarm_init=0;
    max_iter=1000;
    terminate=0;
    crit=1e-9; 
    iter=1;
    while (terminate==0 & iter<max_iter)
        k1m = function1_k1m(zbarm_init);
        k0m = function1_k0m(zbarm_init,k1m);
        A1m = function1_A1m(psi,k1m,rho,phi);
        A0m = function1_A0m(beta,mu,psi,k0,k1,theta,sigma,A0,A1,k0m,k1m,A1m,mu_d,phi_e,phi_d);
        zbarm = A0m; % Unconditional expectation of z_{m,t}
        if abs(zbarm-zbarm_init)<crit
            terminate=1;
        end        
        zbarm_init=zbarm;
        iter=iter+1;
    end
end

[k0m,k1m,A0m,A1m,zbarm] = solve_param_div1(beta,mu,psi,theta,rho,phi_e,sigma, ...
        phi,k0,k1,A0,A1,phi_d,mu_d);

%% TAUCHEN FUNCTION
function [prob] = cdf_standard_normal(x)
    prob=0.5*erfc(-x/sqrt(2));
end    

function [yi,Pi] = DAF_tauchen_normal(mu,lambda,sigma_epsilon,N)
    m=3; % Standard for a normal r.v. (covers 99.7% of the distribution)
    yi=zeros(N,1);
    Pi=zeros(N,N);
    mu_y=mu/(1-lambda); % Mean of an AR(1) 
    sigma_y=((sigma_epsilon^2)/(1-lambda^2))^0.5; % Std of an AR(1) 
    yi(1)=mu_y-m*sigma_y;
    yi(N)=mu_y+m*sigma_y;
    w=(yi(N)-yi(1))/(N-1); % Distance between each gridpoint (equispaced)
    for n=2:N-1
        yi(n)=yi(n-1)+w; 
    end    
    
    % Follow Tauchen (1986):
    for j=1:N
        for k=1:N
            if k==1
                Pi(j,k)=cdf_standard_normal((yi(k)-mu-lambda*yi(j)+w/2)/sigma_epsilon);
            elseif k == N
                Pi(j,k)=1-cdf_standard_normal((yi(k)-mu-lambda*yi(j)-w/2)/sigma_epsilon);
            else
                Pi(j,k)=cdf_standard_normal((yi(k)-mu-lambda*yi(j)+w/2)/sigma_epsilon)- ...
                    cdf_standard_normal((yi(k)-mu-lambda*yi(j)-w/2)/sigma_epsilon);
            end
        end
    end
end