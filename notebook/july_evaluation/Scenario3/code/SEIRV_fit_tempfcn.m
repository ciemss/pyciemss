%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% This code carries out data fitting for the second wave using the SEIRV
% model. 
%
% NOTE: High titer -> Mean half-life 0.99 days
%       Low titer  -> Mean half-life 7.9 days
%       These values impact tau0 in the getDecay() function.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

rng('default');

clear;clc
set(groot, 'defaultAxesTickLabelInterpreter','latex');
set(groot, 'defaultLegendInterpreter','latex');
set(0,'defaulttextInterpreter','latex','defaultAxesFontSize',16) 
format long

bl = '#0072BD';
br = '#D95319';

%% Load data
load('data.mat')

V = cRNA2.*F2;
split = 78; %split 1 week after first day of vaccine (12/11/2020)
V = V(1:split);
tspan = 1:length(V);

%% Curve-fitting
options = optimoptions('fmincon','TolX',1e-12,'TolFun',1e-12,'MaxIter',50000,'MaxFunEvals',100000,'display','off');

  beta_fixed = 4.48526e7;
    lb = [0     51    beta_fixed  10 ];
    ub = [1E-4  796   beta_fixed  5000];
    p0 = [9.06e-08 360 beta_fixed 1182];

% 
%% try Global Search
gs = GlobalSearch;
ms = MultiStart('Display','iter');

problem = createOptimProblem('fmincon','x0',p0,...
    'objective',@(param)obj_fun(param,tspan,V),'lb',lb,'ub',ub);

[best_params,SSE] = run(ms,problem,25);

parameter = ["lambda";"alpha";"beta";"E(0)";"SSE"];
estimated_val = [best_params';SSE];
t = table(parameter,estimated_val)

%% Simulate with best params

alpha = best_params(2);
beta = best_params(3);

traveltime = 18; % hours
k = getDecay(1); % use first time point

eta = 1 - exp(-k*traveltime);

% total population served by DITP
N0 = 2300000;

E0 = best_params(4);
I0 = V(1)/(alpha*beta*(1-eta));
R0 = 0;
S0 = N0 - (E0 + I0 + R0);
V0 = V(1); % use first data point 
ICs  = [S0 E0 I0 R0 V0 E0];

[T,Y] = ode45(@SEIRV,1:length(cRNA2),ICs,[],best_params);


%% Plot
time = datetime(2020,9,30) + caldays(0:length(cRNA2)-1);

figure()
    t = tiledlayout(1,2);

    nexttile
    plot(time2(2:end),log10(diff(Y(:,5))),'LineWidth',2); hold on
    plot(time2(2:end),log10(cRNA2(2:end).*F2(2:end)),'.','markersize',20,'LineWidth',2,'Color',br);
    ylabel('$\log_{10}$ viral RNA copies')
    xline(split,'--','LineWidth',2,'Color',[0 1 0])
    ylim([13.5 inf])
    xlim([time(18-1) time(116-1)])


    nexttile
    plot(time(2:end),log10(diff(Y(:,6))),'LineWidth',2); hold on
    p2 = plot(time2(2:end),log10(newRepCases2(2:end)),'LineWidth',2,'Color',br);
    ylabel('$\log_{10}$ Daily Incidence');

    [max1,index1] = max(diff(Y(:,6))); %simulation max
    xline(time2(index1+1),'--','LineWidth',2,'Color',bl)
    [max2,index2] = max(newRepCases2); %simulation max
    xline(time2(index2),'--','LineWidth',2,'Color',br)

    legend('Model','Data','Location','NorthWest')
    ylim([2.379 4.5])
    xlim([time2(2) time2(118)])

    hold off
    
    %%
    
    f = gcf;
    exportgraphics(f,'fitting_with_temperature.pdf','Resolution',600)

    %%
    
    figure
    box on; hold on;

    %estimate R
    y = (diff(Y(:,6)));
    x = (newRepCases2(2:end));
    X = [ones(length(x),1) x];
    b = X\y;

    yCalc2 = X*b;%b1*x;
    scatter(x,y,20,'k','LineWidth',2);
    plot(x,yCalc2,'r','LineWidth',2)
    ylim([0 inf])

    ylabel('Predicted cases');
    xlabel('Reported cases')

    %calculate R2
    Rsq2 = 1 - sum((y-yCalc2).^2)/sum((y-mean(y)).^2);

    R = corrcoef(x,y); 

    f = gcf;
    exportgraphics(f,'corr_1.pdf','Resolution',600)
%% functions

function err = obj_fun(param,tspan,data)
    traveltime = 18;% hours
    k = getDecay(1); % use first time point

    eta = 1 - exp(-k*traveltime);

    % total population served by DITP
    N0 = 2300000;

    E0 = param(4);
    I0 = data(1)/(param(2)*param(3)*(1-eta));
    R0 = 0;
    S0 = N0 - (E0 + I0 + R0);
    V0 = data(1);                
    ICs  = [S0 E0 I0 R0 V0 E0];

    [~,Y] = ode45(@SEIRV,tspan,ICs,[],param(1:4));

    % get daily virus
    cumVirus = Y(:,5);
    dailyVirus = diff(cumVirus);

    temp = log10(data(2:end)) - log10(abs(dailyVirus));
    adiff = rmmissing(temp);

    err = sum((adiff).^2);
end

function k = getDecay(t)
    % compute temperature-adjusted decay rate of viral RNA
    
    % high titer -> tau0 = 0.99 days * 24 hours/day = 23.76
    % low titer  -> tau0 = 7.9 days * 24 hours/day  = 189.6

    tau0 = 189.6;%23.76;
    Q0 = 2.5;
    T0 = 20;

    % get current temperature using best-fit sine function
    A = 3.624836409841919;
    B = 0.020222716119084;
    C = 4.466530666828714;
    D = 16.229757918464635;

    T = A*sin(B*t - C) + D;

    tau = tau0*Q0.^(-(T - T0)/10);

    k = log(2)./tau;

end




function dy = SEIRV(t,y,param)
    % parameters to be fit
    lambda = param(1);
    alpha = param(2);
    beta = param(3);

    dy = zeros(6,1);
    
    S = y(1);  
    E = y(2);      
    I = y(3);
    R = y(4);
    V = y(5);

    traveltime = 18; % hours
    k = getDecay(t);

    eta = 1 - exp(-k*traveltime);

    sigma = 1/3;
    gamma = 1/8;
    

    dy(1) = -lambda*S*I;
    dy(2) = lambda*S*I - sigma*E;                               
    dy(3) = sigma*E - gamma*I;
    dy(4) = gamma*I;
    dy(5) = alpha*beta*(1-eta)*I;
    dy(6) = lambda*S*I;       % track cumulative cases
end