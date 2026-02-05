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
        max(abs(V1_VC(:)-V0(:)))
        it
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
        max(abs(V1_PC(:)-V0(:)))
        it
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
        max(abs(V1_PD(:)-V0(:)))
        it
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