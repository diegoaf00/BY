%DIEGO ALVAREZ FLORES
%BY REPLICATION: TWO STATE VARIABLES
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
sigma_w=0.23e-5;
theta=(1-gamma)/(1-1/psi);
nu1=0.987;
phi=3;
mu_d=0.0015;
phi_d=4.5;

%% TAUCHEN FUNCTION
Nx=50; % Number of grid points for x_t
Ns2=50; % Number of grid points for sigma_t^2

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

[sigma2_grid,S_matrix] = DAF_tauchen_normal((1-nu1)*sigma^2,nu1,sigma_w,Ns2); % Get grid and transition matrix for sigma_t^2

x_grid=cell(Ns2,1); % Pre-define the x-grid
P_matrix=cell(Ns2,1);% Pre-define the transition probability matrix for each point in sigma_t^2
for j=1:Ns2
    [x_grid{j},P_matrix{j}]=DAF_tauchen_normal(0,rho,phi_e*sqrt(sigma2_grid(j)),Nx); % Get x_grid and P_matrix for each point in sigma_t^2
end
%% VFI: POLICY FUNCTIONS

% I have j as current sigma^2, i as current x. 
% For that i, there are transition probabilities that sum up to one defined on the j grid of x. 
% If I want to take expectations over sigma^2 and x at t+1, the transition probability matrix of sigma gives me the probability of ending up at each k for sigma^2 in t+1. 
% If k != j, the grid for x changes to k, but I have transition probabilities for x defined over j. 
% Hence, I have to use those probabilities. 
% Since the k grid is different, I interpolate the value of the value function to match grid j.
function [V1_VC] = value_function_iteration_VC2(Nx,beta,gamma,theta,mu,x_grid,sigma2_grid,P_matrix,max_iter,crit,Ns2,S_matrix)
    V0=ones(Nx,Ns2); % Initial value function
    V1_VC=ones(Nx,Ns2); % Initial value function after iteration i
    terminate=0;
    it=1;
    while (terminate==0 & it<max_iter)
        for j=1:Ns2
            V_interpolate=zeros(Nx,Ns2); % I have to interpolate over each grid, pre-define the matrix
            for k=1:Ns2
                x_winsor=min(max(x_grid{j},min(x_grid{k})),max(x_grid{k})); % I make sure that x is bewteen x_max and x_min from the Tauchen discretization
                V_interpolate(:,k)=interp1(x_grid{k},(V0(:,k)).^(1-gamma),x_winsor); % Interpolate V for the expectation
            end    
            V1_VC(:,j)=((1-beta)+beta.*exp(((1-gamma)/theta).*(mu+x_grid{j})+(0.5/theta)*(1-gamma)^2*sigma2_grid(j)).*...
                ((P_matrix{j}*V_interpolate)*(S_matrix(j,:)') ).^(1/theta)).^(theta/(1-gamma)); % See pdf
        end
        % Check for convergence
        if max(abs(V1_VC(:)-V0(:)))<crit
            terminate=1;
        end        
        V0=V1_VC;
        it=it+1;
    end 
end

function [V1_PC] = value_function_iteration_PC2(Nx,beta,psi,theta,mu,x_grid,sigma2_grid,P_matrix,max_iter,crit,Ns2,S_matrix)
    V0=ones(Nx,Ns2); % Initial value function
    V1_PC=ones(Nx,Ns2); % Initial value function after iteration i
    terminate=0;
    it=1;
    while (terminate==0 & it<max_iter)
         for j=1:Ns2
            V_interpolate=zeros(Nx,Ns2); % I have to interpolate over each grid, pre-define the matrix
            for k=1:Ns2
                x_winsor=min(max(x_grid{j},min(x_grid{k})),max(x_grid{k})); % I make sure that x is bewteen x_max and x_min from the Tauchen discretization
                V_interpolate(:,k)=interp1(x_grid{k},(1+V0(:,k)).^theta,x_winsor); % Interpolate V for the expectation
            end
            V1_PC(:,j)=beta.*exp((1-1/psi).*(mu+x_grid{j})+0.5*theta*(1-1/psi)^2*sigma2_grid(j)).*((P_matrix{j}*V_interpolate)*(S_matrix(j,:)')).^(1/theta); % See pdf 
         end
        % Check for convergence
        if max(abs(V1_PC(:)-V0(:)))<crit
            terminate=1;
        end        
        V0=V1_PC;
        it=it+1;
    end 
end

function [V1_PD] = value_function_iteration_PD2(Nx,beta,psi,theta,mu,x_grid,sigma2_grid,P_matrix,max_iter,crit,mu_d,phi,phi_d,PC,Ns2,S_matrix)
    V0=ones(Nx,Ns2); % Initial value function
    V1_PD=ones(Nx,Ns2); % Initial value function after iteration i
    terminate=0;
    it=1;
    while (terminate==0 & it<max_iter)
        for j=1:Ns2
            V_interpolate=zeros(Nx,Ns2); % I have to interpolate over each grid, pre-define the matrix
            for k=1:Ns2
                x_winsor=min(max(x_grid{j},min(x_grid{k})),max(x_grid{k})); % I make sure that x is bewteen x_max and x_min from the Tauchen discretization
                V_interpolate(:,k)=interp1(x_grid{k},((1+PC(:,k)).^(theta-1)).*(1+V0(:,k)),x_winsor); % Interpolate V for the expectation
            end
            V1_PD(:,j)=beta^theta.*exp((theta-1-theta/psi).*(mu+x_grid{j})+mu_d+phi*x_grid{j}+0.5*(theta-1-theta/psi)^2*sigma2_grid(j)+0.5*phi_d^2*sigma2_grid(j)).* ...
                (PC(:,j)).^(1-theta).*(P_matrix{j}*V_interpolate)*(S_matrix(j,:)'); % See pdf 
        end
        % Check for convergence
        if max(abs(V1_PD(:)-V0(:)))<crit
            terminate=1;
        end        
        V0=V1_PD;
        it=it+1;
    end 
end


max_iter=10000;
crit=1e-5; % Change to 1e-5 for precision

[V1_VC] = value_function_iteration_VC2(Nx,beta,gamma,theta,mu,x_grid,sigma2_grid,P_matrix,max_iter,crit,Ns2,S_matrix);
[V1_PC] = value_function_iteration_PC2(Nx,beta,psi,theta,mu,x_grid,sigma2_grid,P_matrix,max_iter,crit,Ns2,S_matrix);
[V1_PD] = value_function_iteration_PD2(Nx,beta,psi,theta,mu,x_grid,sigma2_grid,P_matrix,max_iter,crit,mu_d,phi,phi_d,V1_PC,Ns2,S_matrix);

X = zeros(Nx, Ns2); % Get an x grid for each sigma grid
for j = 1:Ns2
    X(:,j) = x_grid{j};
end
[S2, ~] = meshgrid(sigma2_grid, 1:Nx); % Pair the S2 grids with the x grids

figure(6) 
surf(X, S2, log(V1_PC)) 
xlabel('x_t'); ylabel('\sigma_t^2'); zlabel('z_t') 
grid on 
title("Log price-consumption ratio") 
saveas(figure(6),'figure6_two.png')

figure(7) 
surf(X, S2, log(V1_PD)) 
xlabel('x_t'); ylabel('\sigma_t^2'); zlabel('z_{m,t}') 
grid on 
title("Log price-dividend ratio") 
saveas(figure(7),'figure7_two.png')

function [Rf] = function2_Rf(Nx,beta,mu,psi,theta,x_grid,sigma2_grid,P_matrix,PC,Ns2,S_matrix)
    Rf=zeros(Nx,Ns2); % Pre-define Rf
    for j=1:Ns2
            V_interpolate=zeros(Nx,Ns2); % I have to interpolate over each grid, pre-define the matrix
            for k=1:Ns2
                x_winsor=min(max(x_grid{j},min(x_grid{k})),max(x_grid{k})); % I make sure that x is bewteen x_max and x_min from the Tauchen discretization
                V_interpolate(:,k)=interp1(x_grid{k},(1+PC(:,k)).^(theta-1),x_winsor); % Interpolate V for the expectation
            end
        Rf(:,j)=(beta^theta.*exp((theta-1-theta/psi).*(mu+x_grid{j})+0.5*(theta-1-theta/psi)^2*sigma2_grid(j)).*(PC(:,j)).^(1-theta).*(P_matrix{j}*V_interpolate)*(S_matrix(j,:))').^(-1)-1; % See pdf
    end    
end

[Rf] = function2_Rf(Nx,beta,mu,psi,theta,x_grid,sigma2_grid,P_matrix,V1_PC,Ns2,S_matrix);

figure(8)
surf(X,S2,log(1+Rf)) 
xlabel('x_t'); ylabel('\sigma_t^2');zlabel('r_f')
grid on
title("Log risk-free rate")
saveas(figure(8),'figure8_two.png')

%% VFI: IRFs
function [x,s2,diff_z,diff_zm] = function2_1_vfi1_irf_z_zm(T,x0,sigma20,rho,phi_e,nu1,sigma,sigma_w, ...
                        x_grid,sigma2_grid,V1_PC,V1_PD,shockType)
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
    
    z=zeros(T+1,1); % Pre-define z
    zm=zeros(T+1,1); % Pre-define zm
    for t=1:T+1
        sigma2_winsor=min(max(s2(t),min(sigma2_grid)),max(sigma2_grid)); % I make sure that sigma^2_t is bewteen sigma^2_max and sigma^2_min from the Tauchen discretization
        PC_over_sigma=zeros(size(sigma2_grid,1),1);
        PD_over_sigma=zeros(size(sigma2_grid,1),1); 
        for j=1:size(sigma2_grid,1)
            x_winsor=min(max(x(t),min(x_grid{j})),max(x_grid{j})); % I make sure that x is bewteen x_max and x_min from the Tauchen discretization for each point in the sigma^2 grid
            PC_over_sigma(j)=interp1(x_grid{j},V1_PC(:,j),x_winsor); % I interpolate the value of the PC ratio over x given j
            PD_over_sigma(j)=interp1(x_grid{j},V1_PD(:,j),x_winsor); % I interpolate the value of the PD ratio over x given j
        end
        exp_z=interp1(sigma2_grid,PC_over_sigma,sigma2_winsor); % I interpolate the value of the PC ratio over sigma^2
        exp_zm=interp1(sigma2_grid,PD_over_sigma,sigma2_winsor); % I interpolate the value of the PD ratio over sigma^2
        z(t)=log(exp_z);
        zm(t)=log(exp_zm);
    end   

    sigma2_winsor_ss=min(max(sigma20,min(sigma2_grid)),max(sigma2_grid)); % I make sure that sigma^2_ss is bewteen sigma^2_max and sigma^2_min from the Tauchen discretization
    PC_over_sigma_ss=zeros(size(sigma2_grid,1),1);
    PD_over_sigma_ss=zeros(size(sigma2_grid,1),1); 
    for j=1:size(sigma2_grid,1)
        x_winsor_ss=min(max(x0,min(x_grid{j})),max(x_grid{j})); % I make sure that x is bewteen x_max and x_min from the Tauchen discretization for each point in the sigma^2 grid
        PC_over_sigma_ss(j)=interp1(x_grid{j},V1_PC(:,j),x_winsor_ss); % I interpolate the value of the PC ratio over x given jss
        PD_over_sigma_ss(j)=interp1(x_grid{j},V1_PD(:,j),x_winsor_ss); % I interpolate the value of the PD ratio over x given jss
    end
    exp_z_ss=interp1(sigma2_grid,PC_over_sigma_ss,sigma2_winsor_ss); % I interpolate the value of the PC ratio over sigma^2
    exp_zm_ss=interp1(sigma2_grid,PD_over_sigma_ss,sigma2_winsor_ss); % I interpolate the value of the PD ratio over sigma^2
    z_ss=log(exp_z_ss);
    zm_ss=log(exp_zm_ss);
    diff_z=z-z_ss;
    diff_zm=zm-zm_ss;
end

T=250;
t=(1:T)';
x_ss=0; % Unconditional expectation of x_t
sigma2_ss=sigma^2; % Unconditional expectation of sigma_t^2

[x_eps,s2_eps,diff_z_eps,diff_zm_eps] = function2_1_vfi1_irf_z_zm(T,x_ss,sigma2_ss,rho,phi_e,nu1,sigma,sigma_w, ...
                   x_grid,sigma2_grid,V1_PC,V1_PD,"epsilon");

[x_omega,s2_omega,diff_z_omega,diff_zm_omega] = function2_1_vfi1_irf_z_zm(T,x_ss,sigma2_ss,rho,phi_e,nu1,sigma,sigma_w, ...
                    x_grid,sigma2_grid,V1_PC,V1_PD,"omega");

figure(9)
plot(t,diff_z_eps(2:end),t,diff_zm_eps(2:end),'LineWidth',2);
grid on;xlabel('t');ylabel('Difference w.r.t. ss');
legend('\Delta z_t (epsilon shock)','\Delta z_{m,t} (epsilon shock)','Location','Best');
title('IRFs to an \epsilon shock');
saveas(figure(9),'figure9_two.png')

figure(10)
plot(t,diff_z_omega(2:end),t,diff_zm_omega(2:end),'LineWidth',2);
grid on;xlabel('t');ylabel('Difference w.r.t. ss');
legend('\Delta z_t (omega shock)','\Delta z_{m,t} (omega shock)','Location','Best');
title('IRFs to an \omega shock');
saveas(figure(10),'figure10_two.png')

%% VFI: TABLE 5 POHL ET AL. (2018)

% POPULATION
function [expected_excess,expected_Rf,sigma_Rm,sigma_Rf,sigma_pd,expected_pd] = function2_simulate_vfi_tauchen_table2(...
    T,burn,rho,phi_e,sigma,mu,mu_d,phi,phi_d,beta,psi,theta,x_grid,V1_PC,V1_PD,P_matrix,nu1,sigma_w,sigma2_grid,S_matrix,Nx,Ns2)
    
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

    z=zeros(T,1); % Pre-define z
    zm=zeros(T,1); % Pre-define zm
    exp_zm=zeros(T,1); % Pre-define exp(zm)
    [Rf]=function2_Rf(Nx,beta,mu,psi,theta,x_grid,sigma2_grid,P_matrix,V1_PC,Ns2,S_matrix); % Get Rf
    Rf_interpol=zeros(T,1); % Pre-define Rf (with interpolation)
    for t=1:T
        sigma2_winsor=min(max(s2(t),min(sigma2_grid)),max(sigma2_grid)); % I make sure that sigma^2_t is bewteen sigma^2_max and sigma^2_min from the Tauchen discretization
        PC_over_sigma=zeros(size(sigma2_grid,1),1);
        PD_over_sigma=zeros(size(sigma2_grid,1),1); 
        Rf_over_sigma=zeros(size(sigma2_grid,1),1);
        for j=1:size(sigma2_grid,1)
            x_winsor=min(max(x(t),min(x_grid{j})),max(x_grid{j})); % I make sure that x is bewteen x_max and x_min from the Tauchen discretization for each point in the sigma^2 grid
            PC_over_sigma(j)=interp1(x_grid{j},V1_PC(:,j),x_winsor); % I interpolate the value of the PC ratio over x given j
            PD_over_sigma(j)=interp1(x_grid{j},V1_PD(:,j),x_winsor); % I interpolate the value of the PD ratio over x given j
            Rf_over_sigma(j)=interp1(x_grid{j},Rf(:,j),x_winsor); % I interpolate the value of Rf over x given j
        end
        exp_z=interp1(sigma2_grid,PC_over_sigma,sigma2_winsor); % I interpolate the value of the PC ratio over sigma^2
        exp_zm(t)=interp1(sigma2_grid,PD_over_sigma,sigma2_winsor); % I interpolate the value of the PD ratio over sigma^2
        Rf_interpol(t)=interp1(sigma2_grid,Rf_over_sigma,sigma2_winsor); % I interpolate the value of Rf over sigma^2
        z(t)=log(exp_z);
        zm(t)=log(exp_zm(t));
    end
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

T=100000;
burn_pop=5000;
rng(seed);

[expected_excess,expected_rf,sigma_rm,sigma_rf,sigma_pd,expected_pd] = function2_simulate_vfi_tauchen_table2(...
    T,burn_pop,rho,phi_e,sigma,mu,mu_d,phi,phi_d,beta,psi,theta,x_grid,V1_PC,V1_PD,P_matrix,nu1,sigma_w,sigma2_grid,S_matrix,Nx,Ns2);

fprintf('TABLE V MOMENTS (VFI (TAUCHEN) SIMULATION: POPULATION)\n');
fprintf('E(Rm-Rf)=%9.3f\n',expected_excess);
fprintf('E(Rf)= %11.3f\n',expected_rf);
fprintf('sigma(Rm)=%8.3f\n',sigma_rm);
fprintf('sigma(Rf)=%8.3f\n',sigma_rf);
fprintf('sigma(p-d)=%7.3f\n',sigma_pd);
fprintf('E(exp(p-d))=%7.3f\n',expected_pd);

% MONTE CARLO
num_sim=500;
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
        function2_simulate_vfi_tauchen_table2( ...
        Tmonths+burn_mc,burn_mc,rho,phi_e,sigma,mu,mu_d,phi,phi_d,beta,psi,theta,x_grid,V1_PC,V1_PD,P_matrix,nu1,sigma_w,sigma2_grid,S_matrix,Nx,Ns2);

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