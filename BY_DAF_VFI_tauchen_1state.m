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

function [Rf] = function1_Rf(beta,mu,psi,theta,x_grid,sigma,P_matrix,PC)
    Rf=(beta^theta.*exp((theta-1-theta/psi).*(mu+x_grid)+0.5*(theta-1-theta/psi)^2*sigma^2).*(PC).^(1-theta).*(P_matrix*((1+PC).^(theta-1)))).^(-1)-1; % See pdf
end

[Rf] = function1_Rf(beta,mu,psi,theta,x_grid,sigma,P_matrix,V1_PC);

figure(8)
plot(x_grid,log(1+Rf),'LineWidth',2)
xlabel('x_t'); ylabel('r_f')
grid on
title("Log risk-free rate")
saveas(figure(8),'figure8_one.png')

%% VFI: IRFs
function [diff_z,diff_zm] = function1_vfi1_irf_z_zm(T,x0,rho,phi_e,sigma,x_grid,V1_PC,V1_PD)
    shockSize=1;
    x=zeros(T+1,1);
    x(1)=x0;
    eps=zeros(T,1);

    eps(1)=shockSize;

    for t=1:T
        x(t+1)=rho*x(t)+phi_e*sigma*eps(t);
    end

    x_winsor=min(max(x,min(x_grid)),max(x_grid)); % I make sure that x is bewteen x_max and x_min from the Tauchen discretization
    exp_z=interp1(x_grid,V1_PC,x_winsor); % I interpolate the value of the PC ratio
    exp_zm=interp1(x_grid,V1_PD,x_winsor); % I interpolate the value of the PD ratio
    z=log(exp_z);
    zm=log(exp_zm);
    
    exp_z_ss=interp1(x_grid,V1_PC,x0); % I interpolate the value of the PC ratio
    exp_zm_ss=interp1(x_grid,V1_PD,x0); % I interpolate the value of the PD ratio
    z_ss=log(exp_z_ss);
    zm_ss=log(exp_zm_ss);
    diff_z=z-z_ss;
    diff_zm=zm-zm_ss;
end

T=250;
t=(1:T)';
x_ss=0; % Unconditional expectation of x_t

[diff1_z_eps,diff1_zm_eps] = function1_vfi1_irf_z_zm(T,x_ss,rho,phi_e,sigma,x_grid,V1_PC,V1_PD);

figure(9)
plot(t,diff1_z_eps(2:end),t,diff1_zm_eps(2:end),'LineWidth',2);
grid on;xlabel('t');ylabel('Difference w.r.t. ss');
legend('\Delta z_t (epsilon shock)','\Delta z_{m,t} (epsilon shock)','Location','Best');
title('IRFs to an \epsilon shock (interpolation)');
saveas(figure(9),'figure9_one.png')


% IRF using Tauchen transition probabilities and the difference in conditional means
function [diff_z,diff_zm] = function1_vfi2_irf_z_zm(T,x0,phi_e,sigma,x_grid,V1_PC,V1_PD,P_matrix)
    shockSize=1;
    [~,index_xss]=min(abs(x_grid-x0)); % Get where is the x closest to x0 in the grid (ss)
    x_shock=x0+phi_e*sigma*shockSize; % Get the value of x right after an epsilon shock (counterfactual)
    [~, index_x_shock]=min(abs(x_grid-x_shock)); % Get where is the x closest to x_shock in the grid
    Nx=length(x_grid);
    prob_t_ss=zeros(Nx,1);
    prob_t_ss(index_xss)=1; % In SS, the model has x=x0 with probability 1 at t=0
    prob_t_shock=zeros(Nx,1);
    prob_t_shock(index_x_shock)=1; % After the shock, the model has x=x_shock with probability 1 at t=0

    expected_ss=zeros(T+1,2);
    expected_shock=zeros(T+1,2);
    for t=0:T
        expected_ss(t+1,:)=[(prob_t_ss')*log(V1_PC),(prob_t_ss')*log(V1_PD)]; % Get the expected value of z, zm in ss path
        expected_shock(t+1,:)=[(prob_t_shock')*log(V1_PC),(prob_t_shock')*log(V1_PD)]; % Get the expected value of z, zm in a "shocked" world
        prob_t_ss=P_matrix'*prob_t_ss; % Update the probabilities using the transition matrix
        prob_t_shock=P_matrix'*prob_t_shock; % Update the probabilities using the transition matrix
    end

    diff_z=expected_shock(:,1)-expected_ss(:,1); % IRF is the difference in the conditional means
    diff_zm=expected_shock(:,2)-expected_ss(:,2); % IRF is the difference in the conditional means
end

[diff2_z_eps,diff2_zm_eps] = function1_vfi2_irf_z_zm(T,x_ss,phi_e,sigma,x_grid,V1_PC,V1_PD,P_matrix);

figure(10)
plot(t,diff2_z_eps(2:end),t,diff2_zm_eps(2:end),'LineWidth',2);
grid on;xlabel('t');ylabel('Difference w.r.t. ss');
legend('\Delta z_t (epsilon shock)','\Delta z_{m,t} (epsilon shock)','Location','Best');
title('IRFs to an \epsilon shock (transition probabilities and expectations)');
saveas(figure(10),'figure10_one.png')