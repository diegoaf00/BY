%DIEGO ALVAREZ FLORES
%BY REPLICATION: ONE STATE VARIABLE
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

%% LOG-LINEARIZATION: POLICY FUNCTIONS

% A simple grid for x_t
function [x_grid] = simple_x_grid(Nx,phi_e,sigma,rho)
    std_x=sqrt((phi_e^2*sigma^2)/(1-rho^2)); % Unconditional standard deviation of x_t
    x_min=-4*std_x;
    x_max=4*std_x;
    x_grid=linspace(x_min,x_max,Nx);
end

function [zt] = function1_log_zt(x_grid,A0,A1)
    zt=A0+A1*x_grid;
end

Nx=50; % Number of grid points for x_t
[x_grid]=simple_x_grid(Nx,phi_e,sigma,rho);
[zt] = function1_log_zt(x_grid,A0,A1);

figure(1)
plot(x_grid,zt,'LineWidth',2)
xlabel('x_t'); ylabel('z_t')
grid on
title("Log price-consumption ratio")
saveas(figure(1),'figure1_one.png')

function [ztm] = function1_log_ztm(x_grid,A0m,A1m)
    ztm=A0m+A1m*x_grid;
end

[ztm] = function1_log_ztm(x_grid,A0m,A1m);

figure(2)
plot(x_grid,ztm,'LineWidth',2)
xlabel('x_t'); ylabel('z_{m,t}')
grid on
title("Log price-dividend ratio")
saveas(figure(2),'figure2_one.png')

function [rf] = function1_rf(beta,mu,psi,theta,k1,A1,x_grid,phi_e,sigma)
    rf=-log(beta)+(1/psi)*mu+(1/psi)*x_grid+ ...
        0.5*(theta-1-theta/(psi^2))*sigma^2+((theta-1)/2)*(k1*A1*phi_e)^2*sigma^2;
end

[rf] = function1_rf(beta,mu,psi,theta,k1,A1,x_grid,phi_e,sigma);

figure(3)
plot(x_grid,rf,'LineWidth',2)
xlabel('x_t'); ylabel('r_f')
grid on
title("Log risk-free rate")
saveas(figure(3),'figure3_one.png')

%% LOG-LINEARIZATION: IRFs
function [x,diff_z,diff_zm] = function1_irf_z_zm(T,x0,rho,phi_e,sigma,A0,A1,A0m,A1m)
    shockSize=1;
    x=zeros(T+1,1);
    x(1)=x0;
    eps=zeros(T,1);

    eps(1)=shockSize;

    for t=1:T
        x(t+1)=rho*x(t)+phi_e*sigma*eps(t);
    end

    z=A0+A1*x;
    zm=A0m+A1m*x;
    z_ss=A0;
    zm_ss=A0m;
    diff_z=z-z_ss;
    diff_zm=zm-zm_ss;
end

T=250;
t=(1:T)';
x_ss=0; % Unconditional expectation of x_t

[x_eps,diff_z_eps,diff_zm_eps] = function1_irf_z_zm(T,x_ss,rho,phi_e,sigma,A0,A1,A0m,A1m);

figure(4)
plot(t,diff_z_eps(2:end),t,diff_zm_eps(2:end),'LineWidth',2);
grid on;xlabel('t');ylabel('Difference w.r.t. ss');
legend('\Delta z_t (epsilon shock)','\Delta z_{m,t} (epsilon shock)','Location','Best');
title('IRFs to an \epsilon shock');
saveas(figure(4),'figure4_one.png')

%% LOG-LINEARIZATION: TABLE 2

% POPULATION
function [expected_excess,expected_Rf,sigma_Rm,sigma_Rf,sigma_pd] = function1_simulate_log_linear_table2(...
    T,burn,rho,phi_e,sigma,mu,mu_d,phi,phi_d,beta,psi,theta,k1,A1,A0m,A1m,k0m,k1m)
    
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

    zm=A0m+A1m*x;
    [rf] = function1_rf(beta,mu,psi,theta,k1,A1,x,phi_e,sigma);
    Rf=exp(rf)-1;

    for t=1:T-1
        rm(t+1)=k0m+k1m*zm(t+1)-zm(t)+gd(t+1);
    end

    Rm=exp(rm)-1;

    t=(burn+1):(T-1);

    Rm_sim=Rm(t);
    Rf_sim=Rf(t);
    zm_sim=zm(t);
    excess_sim=Rm_sim-Rf_sim;

    expected_excess=mean(excess_sim)*12*100; % I annualize returns by multiplying by 12
    expected_Rf=mean(Rf_sim)*12*100;
    sigma_Rm=std(Rm_sim)*sqrt(12)*100; % I annualize volatility by multiplying by sqrt(12)
    sigma_Rf=std(Rf_sim)*sqrt(12)*100;
    sigma_pd=std(zm_sim);
end

T=2000000;
burn_pop=100000;
rng(seed);

[expected_excess,expected_rf,sigma_rm,sigma_rf,sigma_pd] = function1_simulate_log_linear_table2(...
    T,burn_pop,rho,phi_e,sigma,mu,mu_d,phi,phi_d,beta,psi,theta,k1,A1,A0m,A1m,k0m,k1m);

fprintf('TABLE II MOMENTS (LOG-LINEAR SIMULATION: POPULATION)\n');
fprintf('E(Rm-Rf)=%9.3f\n',expected_excess);
fprintf('E(Rf)= %11.3f\n',expected_rf);
fprintf('sigma(Rm)=%8.3f\n',sigma_rm);
fprintf('sigma(Rf)=%8.3f\n',sigma_rf);
fprintf('sigma(p-d)=%7.3f\n',sigma_pd);

% MONTE CARLO
num_sim=1000;
Tmonths=840;
burn_mc=100;

expected_excess_mc=zeros(num_sim,1);
expected_Rf_mc=zeros(num_sim,1);
sigma_Rm_mc=zeros(num_sim,1);
sigma_Rf_mc=zeros(num_sim,1);
sigma_pd_mc=zeros(num_sim,1);

for k=1:num_sim
    [expected_excess,expected_Rf,sigma_Rm,sigma_Rf,sigma_pd]= ...
        function1_simulate_log_linear_table2( ...
        Tmonths+burn_mc,burn_mc,rho,phi_e,sigma,mu,mu_d,phi,phi_d,beta,psi,theta,k1,A1,A0m,A1m,k0m,k1m);

    expected_excess_mc(k)=expected_excess;
    expected_Rf_mc(k)=expected_Rf;
    sigma_Rm_mc(k)=sigma_Rm;
    sigma_Rf_mc(k)=sigma_Rf;
    sigma_pd_mc(k)=sigma_pd;
end

mean_expected_excess_mc=mean(expected_excess_mc);
mean_expected_rf_mc=mean(expected_Rf_mc);
mean_sigma_rm_mc=mean(sigma_Rm_mc);
mean_sigma_rf_mc=mean(sigma_Rf_mc);
mean_sigma_pd_mc=mean(sigma_pd_mc);

fprintf('TABLE II MOMENTS (LOG-LINEAR SIMULATION: MONTE CARLO)\n');
fprintf('E(Rm-Rf)=%10.3f\n',mean_expected_excess_mc);
fprintf('E(Rf)= %12.3f\n',mean_expected_rf_mc);
fprintf('sigma(Rm)=%9.3f\n',mean_sigma_rm_mc);
fprintf('sigma(Rf)=%9.3f\n',mean_sigma_rf_mc);
fprintf('sigma(p-d)=%8.3f\n',mean_sigma_pd_mc);

%% LOG-LINEARIZATION: TABLE 1
%{
function [acf_k] = function_acf_k(g,k)
    T=length(g);
    g0=g(1:T-k)-mean(g);
    g1=g(1+k:T)-mean(g);
    acf_k=(g0'*g1)/sqrt((g0'*g0)*(g1'*g1));
end

function [var_ratio] = function_var_ratio(g,k) % Variance Ratio from Lo and MacKinlay (1988, 1989)
    T=length(g);
    Var=var(g,1);
    Sum=zeros(T-k+1,1);
    for t=1:T-k+1
        Sum(t)=sum(g(t:t+k-1));
    end
    var_ratio=var(Sum,1)/(k*Var);
end

function [ga] = function_monthly_to_annual(gm)
    Tm=length(gm);
    Ta=floor(Tm/12);
    gm=gm(1:12*Ta); % Get full years only
    ga=reshape(gm,12,Ta);
    ga=sum(ga,1).'; % Sum log growth
end

function[g,gd] = function1_sim_monthly_growth(T,rho,phi_e,sigma,mu,mu_d,phi,phi_d)
    epsilon=randn(T,1);
    eta=randn(T,1);
    u=randn(T,1);

    x=zeros(T,1);
    x(1)=0;

    g=zeros(T,1);
    gd=zeros(T,1);

    for t=1:T-1
        g(t+1)=mu+x(t)+sigma*eta(t+1);
        gd(t+1)=mu_d+phi*x(t)+phi_d*sigma*u(t+1);
        x(t+1)=rho*x(t)+phi_e*sigma*epsilon(t+1);
    end
end

function [sigma_g,AC1,AC2,AC5,AC10,VR2,VR5,VR10] = function1_moments_cons_table1(ga)
    sigma_g=100*std(ga,1);

    AC1=function_acf_k(ga,1);
    AC2=function_acf_k(ga,2);
    AC5=function_acf_k(ga,5);
    AC10=function_acf_k(ga,10);

    VR2=function_var_ratio(ga,2);
    VR5=function_var_ratio(ga,5);
    VR10=function_var_ratio(ga,10);
end

function [sigma_gd,AC1_gd] = function1_moments_div_table1(gda)
    sigma_gd=100*std(gda,1);
    AC1_gd=function_acf_k(gda,1);
end

rng(seed);
num_sim=1000;
Tmonths=840;
burn=100;
pop_months=2000000;

consumption_stats=zeros(num_sim,8);
dividend_stats=zeros(num_sim,2);
corr_div_cons=zeros(num_sim,1);

for k = 1:num_sim
    [g_m,gd_m] = function1_sim_monthly_growth(Tmonths+burn,rho,phi_e,sigma,mu,mu_d,phi,phi_d);
    g_m=g_m(burn+1:end);
    gd_m=gd_m(burn+1:end);

    [g_a] = function_monthly_to_annual(g_m);
    [gd_a] = function_monthly_to_annual(gd_m);

    [sig,ac1,ac2,ac5,ac10,vr2,vr5,vr10] = function1_moments_cons_table1(g_a);
    consumption_stats(k,:) = [sig,ac1,ac2,ac5,ac10,vr2,vr5,vr10];
    [sigd,ac1d] = function1_moments_div_table1(gd_a);
    dividend_stats(k,:) = [sigd,ac1d];
    corr_div_cons(k)=corr(g_a,gd_a);
end

[g_pop_m,gd_pop_m] = function1_sim_monthly_growth(pop_months,rho,phi_e,sigma,mu,mu_d,phi,phi_d);

[g_pop_a] = function_monthly_to_annual(g_pop_m);
[gd_pop_a] = function_monthly_to_annual(gd_pop_m);

[sig,ac1,ac2,ac5,ac10,vr2,vr5,vr10] = function1_moments_cons_table1(g_pop_a);
consumption_stats_pop = [sig,ac1,ac2,ac5,ac10,vr2,vr5,vr10];
[sigd,ac1d] = function1_moments_div_table1(gd_pop_a);
dividend_stats_pop = [sigd,ac1d];
corr_div_cons_pop=corr(g_pop_a,gd_pop_a);

summary=@(X)[mean(X,1); prctile(X,95,1); prctile(X,5,1)];
G_summary=summary(consumption_stats);
GD_summary=summary(dividend_stats);
CORR_summary=[mean(corr_div_cons);prctile(corr_div_cons,95);prctile(corr_div_cons,5)];

% Table format:
row_names_g = {'sigma(g)','AC(1)','AC(2)','AC(5)','AC(10)','VR(2)','VR(5)','VR(10)'};
row_names_gd = {'sigma(g_d)','AC(1)'};

Table_g=array2table([G_summary.' consumption_stats_pop(:)], ...
      'VariableNames',{'Mean','P95','P05','Pop'},'RowNames',row_names_g);
Table_gd=array2table([GD_summary.' dividend_stats_pop(:)], ...
      'VariableNames',{'Mean','P95','P05','Pop'},'RowNames',row_names_gd);
Table_corr=table([CORR_summary; corr_div_cons_pop], ...
    'VariableNames',{'Value'},'RowNames',{'corr(g,gd)_Mean','corr(g,gd)_P95','corr(g,gd)_P05','corr(g,gd)_Pop'});
disp('BY Table I (2 state variables): Consumption growth'); disp(Table_g);
disp('BY Table I (2 state variables): Dividend growth'); disp(Table_gd);
disp('BY Table I (2 state variables): corr(g,g_d)');disp(Table_corr);
%}