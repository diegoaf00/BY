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

%% TAUCHEN FUNCTION
Nx=50; % Number of grid points

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

[x_grid,P_matrix] = DAF_tauchen_normal(0,rho,phi_e*sigma,Nx); % Get grid and transition matrix

%% VFI: POLICY FUNCTIONS
function [V1_VC] = value_function_iteration_VC1(Nx,beta,gamma,theta,mu,x_grid,sigma,P_matrix,max_iter,crit)
    V0=ones(Nx,1); % Initial value function
    terminate=0;
    it=1;
    while (terminate==0 & it<max_iter)
        V1_VC=((1-beta)+beta.*exp(((1-gamma)/theta).*(mu+x_grid)+(0.5/theta)*(1-gamma)^2*sigma^2).* (P_matrix*(V0.^(1-gamma))).^(1/theta)).^(theta/(1-gamma)); % See pdf 
        % Check for convergence
        if max(abs(V1_VC-V0))<crit
            terminate=1;
        end        
        V0=V1_VC;
        it=it+1;
    end 
end

function [V1_PC] = value_function_iteration_PC1(Nx,beta,psi,theta,mu,x_grid,sigma,P_matrix,max_iter,crit)
    V0=ones(Nx,1); % Initial value function
    terminate=0;
    it=1;
    while (terminate==0 & it<max_iter)
        V1_PC=beta.*exp((1-1/psi).*(mu+x_grid)+0.5*theta*(1-1/psi)^2*sigma^2).*(P_matrix*(1+V0).^theta).^(1/theta); % See pdf 
        % Check for convergence
        if max(abs(V1_PC-V0))<crit
            terminate=1;
        end        
        V0=V1_PC;
        it=it+1;
    end 
end

function [V1_PD] = value_function_iteration_PD1(Nx,beta,psi,theta,mu,x_grid,sigma,P_matrix,max_iter,crit,mu_d,phi,phi_d,PC)
    V0=ones(Nx,1); % Initial value function
    terminate=0;
    it=1;
    while (terminate==0 & it<max_iter)
        V1_PD=beta^theta.*exp((theta-1-theta/psi).*(mu+x_grid)+mu_d+phi*x_grid+0.5*(theta-1-theta/psi)^2*sigma^2+0.5*phi_d^2*sigma^2).* ...
            (PC).^(1-theta).*(P_matrix*(((1+PC).^(theta-1)).*(1+V0))); % See pdf 
        % Check for convergence
        if max(abs(V1_PD-V0))<crit
            terminate=1;
        end        
        V0=V1_PD;
        it=it+1;
    end 
end

max_iter=10000;
crit=1e-9;

[V1_VC] = value_function_iteration_VC1(Nx,beta,gamma,theta,mu,x_grid,sigma,P_matrix,max_iter,crit);
[V1_PC] = value_function_iteration_PC1(Nx,beta,psi,theta,mu,x_grid,sigma,P_matrix,max_iter,crit);
[V1_PD] = value_function_iteration_PD1(Nx,beta,psi,theta,mu,x_grid,sigma,P_matrix,max_iter,crit,mu_d,phi,phi_d,V1_PC);

figure(6)
plot(x_grid,log(V1_PC),'LineWidth',2)
xlabel('x_t'); ylabel('z_t')
grid on
title("Log price-consumption ratio")
saveas(figure(6),'figure6_one.png')

figure(7)
plot(x_grid,log(V1_PD),'LineWidth',2)
xlabel('x_t'); ylabel('z_{m,t}')
grid on
title("Log price-dividend ratio")
saveas(figure(7),'figure7_one.png')

