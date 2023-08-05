%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% This file fits a sinusoidal function to the wastewater temperature data.
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

xdata = 1:length(tempData2(1:end-1));
ydata = tempData2(1:end-1)';

% fit sinusoidal model f(x) = A*sin(Bx - C) + D
fun = @(p,xdata)(p(1)*sin(p(2)*xdata - p(3)) + p(4));

%     A    B      C   D
p0 = [4    0.02   0   15];
[params,sse] = lsqcurvefit(fun,p0,xdata,ydata);

disp(params)

figure()
    plot(flowDate2(1:end-1),ydata,'.','MarkerSize',20,'Color',br); hold on
    plot(xdata,fun(params,xdata),'LineWidth',4,'Color',bl); 
    xlim tight
    ylabel('Temperature ($^\circ$C)')
