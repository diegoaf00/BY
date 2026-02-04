%DIEGO ALVAREZ FLORES
%BY REPLICATION: ONE STATE VARIABLE
%VFI: TAUCHEN
%FEBRUARY 2026

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

%% VFI: TABLE 5 POHL ET AL. (2018)

% POPULATION
function [expected_excess,expected_Rf,sigma_Rm,sigma_Rf,sigma_pd,expected_pd] = function1_simulate_vfi_tauchen_table2(...
    T,burn,rho,phi_e,sigma,mu,mu_d,phi,phi_d,beta,psi,theta,x_grid,V1_PC,V1_PD,P_matrix)
    
    x=zeros(T,1);
    g=zeros(T,1);
    gd=zeros(T,1);
    rm=zeros(T,1);

    epsilon=randn(T,1);
    eta=randn(T,1);
    u=randn(T,1);

    x(1)=0; % Unconditional expectation of x_t

    for t=1:T-1
        x(t+1)=rho*x(t)+phi_e*sigma*epsilon(t+1);
        g(t+1)=mu+x(t)+sigma*eta(t+1);
        gd(t+1)=mu_d+phi*x(t)+phi_d*sigma*u(t+1);
    end

    x_winsor=min(max(x,min(x_grid)),max(x_grid)); % I make sure that x is bewteen x_max and x_min from the Tauchen discretization
    exp_z=interp1(x_grid,V1_PC,x_winsor); % I interpolate the value of the PC ratio
    exp_zm=interp1(x_grid,V1_PD,x_winsor); % I interpolate the value of the PD ratio
    z=log(exp_z);
    zm=log(exp_zm);

    [Rf]=function1_Rf(beta,mu,psi,theta,x_grid,sigma,P_matrix,V1_PC);
    Rf_interpol=interp1(x_grid,Rf,x_winsor); % I interpolate the value of the risk-free rate
    rf=log(1+Rf_interpol);

    for t=1:T-1
        rm(t+1)=log(exp_zm(t+1) + 1)-log(exp_zm(t))+gd(t+1); % log market return
    end

    t=(burn+1):(T-1);

    rm_sim=rm(t);
    rf_sim=rf(t);
    zm_sim=zm(t);
    gd_sim=gd(t);

    T_months=length(rm_sim);
    years=floor(T_months/12); % Number of full years
    t_years=1:(12*years);

    rm_reshape=reshape(rm_sim(t_years),12,years);
    rf_reshape=reshape(rf_sim(t_years),12,years);
    gd_reshape=reshape(gd_sim(t_years),12,years);

    rm_annual=sum(rm_reshape,1)'; % Get annual log returns
    rf_annual=sum(rf_reshape,1)'; % Get annual log returns

    Rm_annual=exp(rm_annual)-1;
    Rf_annual=exp(rf_annual)-1;

    excess_annual=Rm_annual-Rf_annual;

    PD_level=exp(zm_sim(t_years));
    PD_reshape=reshape(PD_level,12,years);
    PD_December=PD_reshape(12,:)'; % Keep one observation per year

    PD_annual=zeros(years,1);

    for i=1:years
        gd_annual=gd_reshape(:,i);
        growth_annual=exp(sum(gd_annual)); % Gross dividend growth over a year
        sum_dividends = sum(exp(cumsum(gd_annual))); % Sum of dividend levels divided by the initial level
        PD_annual(i)=PD_December(i)*growth_annual/sum_dividends; % Price divided by sum of dividends
    end

    pd_annual=log(PD_annual);

    expected_excess=mean(excess_annual)*100;
    expected_Rf=mean(Rf_annual)*100;
    sigma_Rm=std(Rm_annual)*100;
    sigma_Rf=std(Rf_annual)*100;

    sigma_pd=std(pd_annual);
    expected_pd=mean(pd_annual);
end

T=2000000;
burn_pop=100000;
rng(seed);

[expected_excess,expected_rf,sigma_rm,sigma_rf,sigma_pd,expected_pd] = function1_simulate_vfi_tauchen_table2(...
    T,burn_pop,rho,phi_e,sigma,mu,mu_d,phi,phi_d,beta,psi,theta,x_grid,V1_PC,V1_PD,P_matrix);

fprintf('TABLE V MOMENTS (VFI (TAUCHEN) SIMULATION: POPULATION)\n');
fprintf('E(Rm-Rf)=%9.3f\n',expected_excess);
fprintf('E(Rf)= %11.3f\n',expected_rf);
fprintf('sigma(Rm)=%8.3f\n',sigma_rm);
fprintf('sigma(Rf)=%8.3f\n',sigma_rf);
fprintf('sigma(p-d)=%7.3f\n',sigma_pd);
fprintf('E(exp(p-d))=%7.3f\n',expected_pd);

% MONTE CARLO
num_sim=1000;
Tmonths=840;
burn_mc=100;

expected_excess_mc=zeros(num_sim,1);
expected_Rf_mc=zeros(num_sim,1);
sigma_Rm_mc=zeros(num_sim,1);
sigma_Rf_mc=zeros(num_sim,1);
sigma_pd_mc=zeros(num_sim,1);
expected_pd_mc=zeros(num_sim,1);

for k=1:num_sim
    [expected_excess,expected_Rf,sigma_Rm,sigma_Rf,sigma_pd,expected_pd]= ...
        function1_simulate_vfi_tauchen_table2( ...
        Tmonths+burn_mc,burn_mc,rho,phi_e,sigma,mu,mu_d,phi,phi_d,beta,psi,theta,x_grid,V1_PC,V1_PD,P_matrix);

    expected_excess_mc(k)=expected_excess;
    expected_Rf_mc(k)=expected_Rf;
    sigma_Rm_mc(k)=sigma_Rm;
    sigma_Rf_mc(k)=sigma_Rf;
    sigma_pd_mc(k)=sigma_pd;
    expected_pd_mc(k)=expected_pd;
end

mean_expected_excess_mc=mean(expected_excess_mc);
mean_expected_rf_mc=mean(expected_Rf_mc);
mean_sigma_rm_mc=mean(sigma_Rm_mc);
mean_sigma_rf_mc=mean(sigma_Rf_mc);
mean_sigma_pd_mc=mean(sigma_pd_mc);
mean_expected_pd_mc=mean(expected_pd_mc);

fprintf('TABLE V MOMENTS (VFI (TAUCHEN) SIMULATION: MONTE CARLO)\n');
fprintf('E(Rm-Rf)=%10.3f\n',mean_expected_excess_mc);
fprintf('E(Rf)= %12.3f\n',mean_expected_rf_mc);
fprintf('sigma(Rm)=%9.3f\n',mean_sigma_rm_mc);
fprintf('sigma(Rf)=%9.3f\n',mean_sigma_rf_mc);
fprintf('sigma(p-d)=%8.3f\n',mean_sigma_pd_mc);
fprintf('E(exp(p-d))=%7.3f\n',mean_expected_pd_mc);