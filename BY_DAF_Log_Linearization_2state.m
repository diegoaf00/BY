%DIEGO ALVAREZ FLORES
%BY REPLICATION: TWO STATE VARIABLES
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
sigma_w=0.23e-5;
theta=(1-gamma)/(1-1/psi);
nu1=0.987;
phi=3;
mu_d=0.0015;
phi_d=4.5;

%% EQUATIONS FOR FIXED POINT METHOD (C0NSUMPTION)
function [k0] = function2_k0(zbar,k1)
    k0=log(exp(zbar)+1)-k1*zbar;
end

function [k1] = function2_k1(zbar)
    k1=exp(zbar)/(exp(zbar)+1);
end

function [A1] = function2_A1(psi,k1,rho)
    A1=(1-1/psi)/(1-k1*rho);
end

function [A2] = function2_A2(theta,psi,k1,A1,phi_e,nu1)
    A2=0.5*((theta-theta/psi)^2+(theta*k1*A1*phi_e)^2)/(theta*(1-k1*nu1));
end

function [A0] = function2_A0(beta,mu,psi,k0,k1,A2,sigma_w,nu1,theta,sigma)
    A0=(log(beta)+mu*(1-1/psi)+k0+k1*A2*sigma^2*(1-nu1)+0.5*sigma_w^2*theta*(k1*A2)^2)/(1-k1);
end

%% FIXED-POINT METHOD (C0NSUMPTION)
function [k0,k1,A0,A1,A2,zbar] = solve_param_cons2(beta,mu,psi,sigma_w,nu1,theta,rho,phi_e,sigma)
    zbar_init=0;
    max_iter=1000;
    terminate=0;
    crit=1e-9; 
    iter=1;
    while (terminate==0 & iter<max_iter)
        k1 = function2_k1(zbar_init);
        k0 = function2_k0(zbar_init,k1);
        A1 = function2_A1(psi,k1,rho);
        A2 = function2_A2(theta,psi,k1,A1,phi_e,nu1);
        A0 = function2_A0(beta,mu,psi,k0,k1,A2,sigma_w,nu1,theta,sigma);
        zbar = A0+A2*sigma^2; % Unconditional expectation of z_t
        if abs(zbar-zbar_init)<crit
            terminate=1;
        end        
        zbar_init=zbar;
        iter=iter+1;
    end
end

[k0,k1,A0,A1,A2,zbar] = solve_param_cons2(beta,mu,psi,sigma_w,nu1,theta,rho,phi_e,sigma);

%% EQUATIONS FOR FIXED POINT METHOD (DIVIDEND)
function [k0m] = function2_k0m(zbarm,k1m)
    k0m=log(exp(zbarm)+1)-k1m*zbarm;
end

function [k1m] = function2_k1m(zbarm)
    k1m=exp(zbarm)/(exp(zbarm)+1);
end

function [A1m] = function2_A1m(psi,k1m,rho,phi)
    A1m=(phi-1/psi)/(1-k1m*rho);
end

function [A2m] = function2_A2m(theta,psi,k1,A1,phi_e,nu1,A2,k1m,A1m,phi_d)
    A2m = (A2*(1-theta)*(1-k1*nu1)+0.5*((theta-1-theta/psi)^2 ...
        +((theta-1)*k1*A1*phi_e+k1m*A1m*phi_e)^2+phi_d^2))/(1-k1m*nu1);
end

function [A0m] = function2_A0m(beta,mu,psi,k0,k1,A2,sigma_w,nu1,theta,sigma,A0,k0m,k1m,A2m,mu_d)
    A0m = (theta*log(beta)+mu*(theta-1-theta/psi)+(theta-1)*(k0+A0*(k1-1))+(theta-1)*k1*A2*sigma^2*(1-nu1) ...
        +k0m+k1m*A2m*sigma^2*(1-nu1)+mu_d+0.5*sigma_w^2*((theta-1)*k1*A2+k1m*A2m)^2)/(1-k1m);
end

%% FIXED-POINT METHOD (DIVIDEND)
function [k0m,k1m,A0m,A1m,A2m,zbarm] = solve_param_div2(beta,mu,psi,sigma_w,nu1,theta,rho,phi_e,sigma, ...
        phi,k0,k1,A0,A1,A2,phi_d,mu_d)
    zbarm_init=0;
    max_iter=1000;
    terminate=0;
    crit=1e-9; 
    iter=1;
    while (terminate==0 & iter<max_iter)
        k1m = function2_k1m(zbarm_init);
        k0m = function2_k0m(zbarm_init,k1m);
        A1m = function2_A1m(psi,k1m,rho,phi);
        A2m = function2_A2m(theta,psi,k1,A1,phi_e,nu1,A2,k1m,A1m,phi_d);
        A0m = function2_A0m(beta,mu,psi,k0,k1,A2,sigma_w,nu1,theta,sigma,A0,k0m,k1m,A2m,mu_d);
        zbarm = A0m+A2m*sigma^2; % Unconditional expectation of z_{m,t}
        if abs(zbarm-zbarm_init)<crit
            terminate=1;
        end        
        zbarm_init=zbarm;
        iter=iter+1;
    end
end

[k0m,k1m,A0m,A1m,A2m,zbarm] = solve_param_div2(beta,mu,psi,sigma_w,nu1,theta,rho,phi_e,sigma, ...
        phi,k0,k1,A0,A1,A2,phi_d,mu_d);

%% LOG-LINEARIZATION: POLICY FUNCTIONS

% A simple grid for sigma^2_t
function [sigma2_grid] = simple_sigma2_grid(Ns2,sigma,sigma_w,nu1)
    std_sigma2=sqrt(sigma_w^2/(1-nu1^2)); % Unconditional standard deviation of sigma^2_t
    sigma2_min=max(1e-9,sigma^2-4*std_sigma2);
    sigma2_max=sigma^2+4*std_sigma2;
    sigma2_grid=linspace(sigma2_min,sigma2_max,Ns2);
end

% A simple grid for x_t
function [x_grid] = simple_x_grid(Nx,phi_e,sigma,rho)
    std_x=sqrt((phi_e^2*sigma^2)/(1-rho^2)); % Unconditional standard deviation of x_t
    x_min=-4*std_x;
    x_max=4*std_x;
    x_grid=linspace(x_min,x_max,Nx);
end

function [zt] = function2_log_zt(sigma2_grid,x_grid,A0,A1,A2)
    [SIGMA2,X]=ndgrid(sigma2_grid,x_grid);
    zt=A0+A1.*X+A2.*SIGMA2;
end

Ns2=50; % Number of grid points for sigma^2_t
Nx=50; % Number of grid points for x_t
[sigma2_grid]=simple_sigma2_grid(Ns2,sigma,sigma_w,nu1);
[x_grid]=simple_x_grid(Nx,phi_e,sigma,rho);
[zt] = function2_log_zt(sigma2_grid,x_grid,A0,A1,A2);

figure(1)
surf(x_grid,sigma2_grid,zt)
xlabel('x_t'); ylabel('\sigma_t^2'); zlabel('z_t')
title("Log price-consumption ratio")
saveas(figure(1),'figure1_two.png')

function [ztm] = function2_log_ztm(sigma2_grid,x_grid,A0m,A1m,A2m)
    [SIGMA2,X]=ndgrid(sigma2_grid,x_grid);
    ztm=A0m+A1m.*X+A2m.*SIGMA2;
end

[ztm] = function2_log_ztm(sigma2_grid,x_grid,A0m,A1m,A2m);

figure(2)
surf(x_grid,sigma2_grid,ztm)
xlabel('x_t'); ylabel('\sigma_t^2'); zlabel('z_{m,t}')
title("Log price-dividend ratio")
saveas(figure(2),'figure2_two.png')

function [rf] = function2_rf(beta,mu,psi,theta,k1,A1,A2,sigma_w,x_grid,sigma2_grid,phi_e)
    [SIGMA2,X]=ndgrid(sigma2_grid,x_grid);
    rf=-log(beta)+(1/psi)*mu+((theta-1)/2)*(k1*A2)^2*sigma_w^2+(1/psi)*X+ ...
        0.5*(theta-1-theta/psi^2)*SIGMA2+((theta-1)/2)*(k1*A1*phi_e)^2*SIGMA2;
end

[rf] = function2_rf(beta,mu,psi,theta,k1,A1,A2,sigma_w,x_grid,sigma2_grid,phi_e);

figure(3)
surf(x_grid,sigma2_grid,rf)
xlabel('x_t'); ylabel('\sigma_t^2'); zlabel('r_f')
title("Log risk-free rate")
saveas(figure(3),'figure3_two.png')

%% LOG-LINEARIZATION: IRFs
function [x,s2,diff_z,diff_zm] = function2_irf_z_zm(T,x0,sigma20,rho,phi_e,nu1,sigma,sigma_w, ...
                        A0,A1,A2,A0m,A1m,A2m,shockType)
    shockSize=1;
    x=zeros(T+1,1);
    s2=zeros(T+1,1);
    x(1)=x0;
    s2(1)=sigma20;
    eps=zeros(T,1);
    omg=zeros(T,1);

    if strcmpi(shockType,'epsilon')
        eps(1)=shockSize;
    elseif strcmpi(shockType,'omega')
        omg(1)=shockSize;
    else
        error('shockType must be ''epsilon'' or ''omega');
    end

    for t=1:T
        x(t+1)=rho*x(t)+phi_e*sqrt(s2(t))*eps(t);
        s2(t+1)=sigma^2+nu1*(s2(t)-sigma^2)+sigma_w*omg(t);
        s2(t+1)=max(s2(t+1),0);
    end

    z=A0+A1.*x+A2.*s2;
    zm=A0m+A1m.*x+A2m.*s2;
    z_ss=A0+A2*sigma^2;
    zm_ss=A0m+A2m*sigma^2;
    diff_z=z-z_ss;
    diff_zm=zm-zm_ss;
end

T=250;
t=(1:T)';
x_ss=0; % Unconditional expectation of x_t
sigma2_ss=sigma^2; % Unconditional expectation of sigma_t^2

[x_eps,s2_eps,diff_z_eps,diff_zm_eps] = function2_irf_z_zm(T,x_ss,sigma2_ss,rho,phi_e,nu1,sigma,sigma_w, ...
                   A0,A1,A2,A0m,A1m,A2m,"epsilon");

[x_omega,s2_omega,diff_z_omega,diff_zm_omega] = function2_irf_z_zm(T,x_ss,sigma2_ss,rho,phi_e,nu1,sigma,sigma_w, ...
                   A0,A1,A2,A0m,A1m,A2m,"omega");

figure(4)
plot(t,diff_z_eps(2:end),t,diff_zm_eps(2:end),'LineWidth',2);
grid on;xlabel('t');ylabel('Difference w.r.t. ss');
legend('\Delta z_t (epsilon shock)','\Delta z_{m,t} (epsilon shock)','Location','Best');
title('IRFs to an \epsilon shock');
saveas(figure(4),'figure4_two.png')

figure(5)
plot(t,diff_z_omega(2:end),t,diff_zm_omega(2:end),'LineWidth',2);
grid on;xlabel('t');ylabel('Difference w.r.t. ss');
legend('\Delta z_t (omega shock)','\Delta z_{m,t} (omega shock)','Location','Best');
title('IRFs to an \omega shock');
saveas(figure(5),'figure5_two.png')

%% LOG-LINEARIZATION: TABLE 5 POHL ET AL. (2018)

% POPULATION
function [expected_excess,expected_Rf,sigma_Rm,sigma_Rf,sigma_pd,expected_pd] = ...
    function2_simulate_log_linear_table4(...
    T,burn,rho,phi_e,sigma,mu,mu_d,phi,phi_d,beta,psi,theta,k1,A1,A0m,A1m,k0m,k1m,nu1,sigma_w,A2,A2m)

    x=zeros(T,1);
    s2=zeros(T,1);
    g=zeros(T,1);
    gd=zeros(T,1);
    rm=zeros(T,1);

    epsilon=randn(T,1);
    eta=randn(T,1);
    u=randn(T,1);
    omega=randn(T,1);

    x(1)=0; % Unconditional expectation of x_t
    s2(1)=sigma^2; % Unconditional expectation of sigma_t^2

    for t=1:T-1
        x(t+1)=rho*x(t)+phi_e*sqrt(s2(t))*epsilon(t+1);
        g(t+1)=mu+x(t)+sqrt(s2(t))*eta(t+1);
        gd(t+1)=mu_d+phi*x(t)+phi_d*sqrt(s2(t))*u(t+1);
        s2(t+1)=sigma^2+nu1*(s2(t)-sigma^2)+sigma_w*omega(t+1);
        s2(t+1)=max(s2(t+1),1e-9);
    end

    zm=A0m+A1m*x+A2m*s2;
    rf=-log(beta)+(1/psi)*mu+((theta-1)/2)*(k1*A2)^2*sigma_w^2+(1/psi)*x+ ...
        0.5*(theta-1-theta/(psi^2))*s2+((theta-1)/2)*(k1*A1*phi_e)^2*s2;

    for t=1:T-1
        rm(t+1)=k0m+k1m*zm(t+1)-zm(t)+gd(t+1);
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

[expected_excess,expected_rf,sigma_rm,sigma_rf,sigma_pd,expected_pd] = ...
    function2_simulate_log_linear_table4(...
    T,burn_pop,rho,phi_e,sigma,mu,mu_d,phi,phi_d,beta,psi,theta,k1,A1,A0m,A1m,k0m,k1m,nu1,sigma_w,A2,A2m);

fprintf('TABLE V MOMENTS (LOG-LINEAR SIMULATION: POPULATION)\n');
fprintf('E(Rm-Rf)=%10.3f\n',expected_excess);
fprintf('E(Rf)= %12.3f\n',expected_rf);
fprintf('sigma(Rm)=%9.3f\n',sigma_rm);
fprintf('sigma(Rf)=%9.3f\n',sigma_rf);
fprintf('sigma(p-d)=%8.3f\n',sigma_pd);
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
        function2_simulate_log_linear_table4( ...
        Tmonths+burn_mc,burn_mc,rho,phi_e,sigma,mu,mu_d,phi,phi_d,beta,psi,theta,k1,A1,A0m,A1m,k0m,k1m,nu1,sigma_w,A2,A2m);

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

fprintf('TABLE V MOMENTS (LOG-LINEAR SIMULATION: MONTE CARLO)\n');
fprintf('E(Rm-Rf)=%10.3f\n',mean_expected_excess_mc);
fprintf('E(Rf)= %12.3f\n',mean_expected_rf_mc);
fprintf('sigma(Rm)=%9.3f\n',mean_sigma_rm_mc);
fprintf('sigma(Rf)=%9.3f\n',mean_sigma_rf_mc);
fprintf('sigma(p-d)=%8.3f\n',mean_sigma_pd_mc);
fprintf('E(exp(p-d))=%7.3f\n',mean_expected_pd_mc);