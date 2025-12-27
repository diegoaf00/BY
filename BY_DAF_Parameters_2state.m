%DIEGO ALVAREZ FLORES
%BY REPLICATION
%FIXED POINT METHOD
%DECEMBER, 2025

clear;
clc;

%% PARAMETERS
sigma=0.0078;
mu=0.0015;
rho=0.979;
psi=1.5;
gamma=10;
beta=0.998;
phi_e=0.044;
sigma_w=0.23e-5;
theta=(1-gamma)/(1-1/psi);
nu1=0.987;
phi=3;
mu_d=0.0015;
phi_d=4.5;

%% EQUATIONS (C0NSUMPTION)
function [k0] = function_k0(zbar,k1)
    k0=log(exp(zbar)+1)-k1*zbar;
end

function [k1] = function_k1(zbar)
    k1=exp(zbar)/(exp(zbar)+1);
end

function [A1] = function_A1(psi,k1,rho)
    A1=(1-1/psi)/(1-k1*rho);
end

function [A2] = function_A2(theta,psi,k1,A1,phi_e,nu1)
    A2=0.5*((theta-theta/psi)^2+(theta*k1*A1*phi_e)^2)/(theta*(1-k1*nu1));
end

function [A0] = function_A0(beta,mu,psi,k0,k1,A2,sigma_w,nu1,theta,sigma)
    A0=(log(beta)+mu*(1-1/psi)+k0+k1*A2*sigma^2*(1-nu1)+0.5*sigma_w^2*theta*(k1*A2)^2)/(1-k1);
end

%% FIXED-POINT (C0NSUMPTION)
function [k0,k1,A0,A1,A2,zbar] = solve_param_cons(beta,mu,psi,sigma_w,nu1,theta,rho,phi_e,sigma)
    zbar_init=0;
    max_iter=1000;
    terminate=0;
    crit=1e-9; 
    iter=1;
    while (terminate==0 & iter<max_iter)
        k1 = function_k1(zbar_init);
        k0 = function_k0(zbar_init,k1);
        A1 = function_A1(psi,k1,rho);
        A2 = function_A2(theta,psi,k1,A1,phi_e,nu1);
        A0 = function_A0(beta,mu,psi,k0,k1,A2,sigma_w,nu1,theta,sigma);
        zbar = A0+A2*sigma^2;
        if abs(zbar-zbar_init)<crit
            terminate=1;
        end        
        zbar_init=zbar;
        iter=iter+1;
    end
end

[k0,k1,A0,A1,A2,zbar] = solve_param_cons(beta,mu,psi,sigma_w,nu1,theta,rho,phi_e,sigma);

%% EQUATIONS (DIVIDEND)
function [k0m] = function_k0m(zbarm,k1m)
    k0m=log(exp(zbarm)+1)-k1m*zbarm;
end

function [k1m] = function_k1m(zbarm)
    k1m=exp(zbarm)/(exp(zbarm)+1);
end

function [A1m] = function_A1m(psi,k1m,rho,phi)
    A1m=(phi-1/psi)/(1-k1m*rho);
end

function [A2m] = function_A2m(theta,psi,k1,A1,phi_e,nu1,A2,k1m,A1m,phi_d)
    A2m = (A2*(1-theta)*(1-k1*nu1)+0.5*((theta-1-theta/psi)^2 ...
        +((theta-1)*k1*A1*phi_e+k1m*A1m*phi_e)^2+phi_d^2))/(1-k1m*nu1);
end

function [A0m] = function_A0m(beta,mu,psi,k0,k1,A2,sigma_w,nu1,theta,sigma,A0,k0m,k1m,A2m,mu_d)
    A0m = (theta*log(beta)+mu*(theta-1-theta/psi)+(theta-1)*(k0+A0*(k1-1))+(theta-1)*k1*A2*sigma^2*(1-nu1) ...
        +k0m+k1m*A2m*sigma^2*(1-nu1)+mu_d+0.5*sigma_w^2*((theta-1)*k1*A2+k1m*A2m)^2)/(1-k1m);
end

%% FIXED-POINT (DIVIDEND)
function [k0m,k1m,A0m,A1m,A2m,zbarm] = solve_param_div(beta,mu,psi,sigma_w,nu1,theta,rho,phi_e,sigma, ...
        phi,k0,k1,A0,A1,A2,phi_d,mu_d)
    zbarm_init=0;
    max_iter=1000;
    terminate=0;
    crit=1e-9; 
    iter=1;
    while (terminate==0 & iter<max_iter)
        k1m = function_k1m(zbarm_init);
        k0m = function_k0m(zbarm_init,k1m);
        A1m = function_A1m(psi,k1m,rho,phi);
        A2m = function_A2m(theta,psi,k1,A1,phi_e,nu1,A2,k1m,A1m,phi_d);
        A0m = function_A0m(beta,mu,psi,k0,k1,A2,sigma_w,nu1,theta,sigma,A0,k0m,k1m,A2m,mu_d);
        zbarm = A0m+A2m*sigma^2;
        if abs(zbarm-zbarm_init)<crit
            terminate=1;
        end        
        zbarm_init=zbarm;
        iter=iter+1;
    end
end

[k0m,k1m,A0m,A1m,A2m,zbarm] = solve_param_div(beta,mu,psi,sigma_w,nu1,theta,rho,phi_e,sigma, ...
        phi,k0,k1,A0,A1,A2,phi_d,mu_d);