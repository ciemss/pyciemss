import matplotlib.pyplot as plt
from matplotlib import ticker
import torch

__all__ = ['setup_ax',
           'plot_predictive',
           'plot_trajectory',
           'plot_intervention_line',
           'sideaxis',
           'sideaxishist']

def setup_ax(ax=None, xlabel='Time (days)', ylabel='Infectious (1000s)'):

    if not ax:
        fig = plt.figure(facecolor='w', figsize=(9, 9))
        ax = fig.add_subplot(111, axisbelow=True)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    return ax

def plot_predictive(datacube, tspan, tmin=None, ax=None, alpha=0.2, color="black", ptiles=[0.05,0.95], vars=["I_obs"], **kwargs):
    if vars[0]=='I_obs':
            datacube['I_obs'] = datacube['I_sol'] + datacube['I_v_sol']    # TODO: This is too specific and needs to be changed
    
    I_low = torch.quantile(datacube[vars[0]], ptiles[0], dim=0).detach().numpy()
    I_up = torch.quantile(datacube[vars[0]], ptiles[1], dim=0).detach().numpy()

    if tmin:
        indeces = tspan >= tmin
    else:
        indeces = torch.ones_like(tspan).bool()

    if not ax:
        fig = plt.figure(facecolor='w')
        ax = fig.add_subplot(111, facecolor='#dddddd', axisbelow=True)
    
    ax.fill_between(tspan[indeces], I_low[indeces], I_up[indeces], alpha=alpha, color=color, **kwargs)

    legend = ax.legend()
    legend.get_frame().set_alpha(0.5)

    return ax

def plot_trajectory(datacube, tspan,  ax=None, color='black', alpha=0.5, lw=0, marker='.', label=None, vars=["I_obs"]):
    if vars[0]=='I_obs':
            datacube['I_obs'] = datacube['I_sol'] + datacube['I_v_sol']    # TODO: This is too specific and needs to be changed
    # Plot the data on three separate curves for S(t), I(t) and R(t)
    if not ax:
        fig = plt.figure(facecolor='w')
        ax = fig.add_subplot(111, facecolor='#dddddd', axisbelow=True)

    ax.plot(tspan, datacube[vars[0]].squeeze().detach().numpy(), color, alpha=alpha, lw=lw, marker=marker, label=label)
    
    return ax

def plot_intervention_line(t, ax=None):
    if not ax:
        fig = plt.figure(facecolor='w')
        ax = fig.add_subplot(111, facecolor='#dddddd', axisbelow=True)

    ylim = ax.get_ylim()

    ax.vlines(t, min(ylim), max(ylim), color="grey", ls='-')

    return ax

def sideaxis(ax):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    return

def sideaxishist(ax):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    # For y-axis
    ax.yaxis.set_major_locator(ticker.NullLocator())
    ax.tick_params(axis='x', labelsize=12)  # change fontsize for x-axis tick labels
    # ax.xaxis.major.formatter._useMathText = True
    return