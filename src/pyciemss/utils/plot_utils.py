import matplotlib.pyplot as plt
from matplotlib import ticker
import torch

__all__ = ['setup_ax',
           'plot_predictive',
           'plot_trajectory',
           'plot_intervention_line',
           'plot_ouu_risk',
           'sideaxis',
           'sideaxishist']

def setup_ax(ax=None, xlabel='Time (days)', ylabel='Infectious', figsize=(10, 6)):
    if not ax:
        fig = plt.figure(facecolor='w', figsize=figsize)
        ax = fig.add_subplot(111, axisbelow=True)
        plt.rc('font', family='serif', size=14.)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    sideaxis(ax)
    return ax

def plot_predictive(datacube, tspan, tmin=None, ax=None, alpha=0.2, color="black", ptiles=[0.05,0.95], vars=["I_obs"], label=None, **kwargs):
    if vars[0] not in datacube and vars[0]=='I_obs':
        datacube['I_obs'] = datacube['I_sol'] + datacube['I_v_sol']    # TODO: This is too specific and needs to be changed
    
    I_low = torch.quantile(datacube[vars[0]], ptiles[0], dim=0).detach().numpy()
    I_up = torch.quantile(datacube[vars[0]], ptiles[1], dim=0).detach().numpy()

    if tmin:
        indeces = tspan >= tmin
    else:
        indeces = torch.ones_like(tspan).bool()
    if not ax:
        ax = setup_ax()
    
    ax.fill_between(tspan[indeces], I_low[indeces], I_up[indeces], alpha=alpha, color=color, label=label, **kwargs)
    if label is not None:
        legend = ax.legend()
        legend.get_frame().set_alpha(0.5)
    return ax

def plot_trajectory(datacube, tspan,  ax=None, color='black', alpha=0.5, lw=0, marker='.', label=None, vars=["I_obs"]):
    if vars[0] not in datacube and vars[0]=='I_obs':
        datacube['I_obs'] = datacube['I_sol'] + datacube['I_v_sol']    # TODO: This is too specific and needs to be changed
    # Plot the data on three separate curves for S(t), I(t) and R(t)
    if not ax:
        ax = setup_ax()

    ax.plot(tspan, datacube[vars[0]].squeeze().detach().numpy(), color, alpha=alpha, lw=lw, marker=marker, label=label)    
    return ax

def plot_intervention_line(t, ax=None):
    if not ax:
        ax = setup_ax()

    ylim = ax.get_ylim()
    ax.vlines(t, min(ylim), max(ylim), color="grey", ls='-')
    return ax

def plot_ouu_risk(datacube, ax=None, xlabel: str='7-day average infectious at 90 days', color: list=['#377eb8', '#ff7f00', '#984ea3', '#ffd92f', '#a65628'], 
                  alpha=0.5, label: list=['alpha-superquantile'], tmin=None):
    if not ax:
        ax = setup_ax()
    ax = plot_predictive(datacube["samples"], torch.tensor(datacube["tspan"]), ax=ax, color='#a65628', ptiles=[0.,1.], alpha=alpha, tmin=tmin)

    bins_hist = 50
    plt.figure()
    plt.rc('font', family='serif', size=14.)
    cax = plt.gca()
    sideaxishist(cax)
    cax.hist(datacube["qoi"], color='#984ea3', bins=bins_hist, histtype='stepfilled', alpha=alpha, density=True)
    miny = min(cax.get_ylim())
    maxy = max(cax.get_ylim())
    for c in range(len(datacube["risk"])):
        cax.vlines(datacube["risk"][c], miny, maxy, linestyle='--', linewidth=2.5, label=label[c], color=color[c])
    cax.set_xlabel(xlabel, size=14)
    cax.legend(loc='upper right', prop={'size': 14})
    return [ax, cax]

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
    return