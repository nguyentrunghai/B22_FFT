
from __future__ import print_function

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.cm

def _regression_line(x, y):
    coeffs = np.polyfit(x, y, 1)
    intercept = coeffs[-1]
    slope = coeffs[-2]
    min_x = np.min(x)
    max_x = np.max(x)
    xl = np.array([min_x, max_x])
    yl = slope*xl + intercept
    line_eq_text = "y = %0.2fx"%slope
    if intercept >= 0:
        line_eq_text += " + "
    else:
        line_eq_text += " - "
    line_eq_text += "%0.2f"%np.abs(intercept)
    return xl, yl, line_eq_text


def _correlation_coef(x, y, repeats=100):
    x = np.array(x)
    y = np.array(y)

    corr_coef = np.corrcoef([x, y] )[0][-1]

    # estimate std
    rs = []
    for repeat in range(repeats):
        sel_ind = np.random.choice(len(x), size=len(x), replace=True)
        sel_x = x[sel_ind]
        sel_y = y[sel_ind]
        rs.append( np.corrcoef([sel_x, sel_y] )[0][-1] )
    std = np.std(rs)
    return corr_coef, std


def _rmse(x,y, repeats=100):
    x = np.array(x)
    y = np.array(y)

    mean_sq_error = (x - y)**2
    mean_sq_error = np.sqrt(mean_sq_error.mean())

    # estimate std
    rmses = []
    for repeat in range(repeats):
        sel_ind = np.random.choice(len(x), size=len(x), replace=True)
        sel_x = x[sel_ind]
        sel_y = y[sel_ind]
        rmse = (sel_x - sel_y)**2 
        rmse = np.sqrt(rmse.mean())
        rmses.append(rmse)

    std = np.std(rmses)
    return mean_sq_error, std


def _armse(x,y, repeats=100):
    x = np.array(x)
    y = np.array(y)
    mean_sq_error = (x - y - (x.mean() - y.mean()) )**2
    mean_sq_error = np.sqrt(mean_sq_error.mean())

    # estimate std
    rmses = []
    for repeat in range(repeats):
        sel_ind = np.random.choice(len(x), size=len(x), replace=True)
        sel_x = x[sel_ind]
        sel_y = y[sel_ind]
        rmse = ( sel_x - sel_y - (sel_x.mean() - sel_y.mean()) )**2
        rmse = np.sqrt(rmse.mean())
        rmses.append(rmse)

    std = np.std(rmses)
    return mean_sq_error, std


def _scatter_plot_info(x, y, label):
    xl, yl, line_eq_text = _regression_line(x, y)
    corr_coef, r_std = _correlation_coef(x, y)
    rmse, rmse_std = _rmse(x,y)
    armse, armse_std = _armse(x,y)

    text = "\n"+label
    text += "\nPearson's R = %0.2f (%0.2f)"%(corr_coef, r_std)
    text += "\nRMSE = %0.2f (%0.2f)"%(rmse, rmse_std)
    text += "\naRMSE = %0.2f (%0.2f)"%(armse, armse_std)
    text += "\n%s\n"%line_eq_text
    return text


def scatter_plot_info(x, y, ligands, out, all_only=False, stdout=False):
    text = _scatter_plot_info(x, y, "All")
    if all_only:
        open(out, "w").write(text)
        return None

    x_ac, y_ac = [], []
    x_inac, y_inac = [], []
    for i, ligand in enumerate(ligands):
        if ".inactive." in ligand or ligand == "phenol.A__AAA":
            x_inac.append(x[i])
            y_inac.append(y[i])
        else:
            x_ac.append(x[i])
            y_ac.append(y[i])

    if len(x_ac) > 0:
        text += _scatter_plot_info(x_ac, y_ac, "active")
    if len(x_inac) > 0:
        text += _scatter_plot_info(x_inac, y_inac, "inactive")

    open(out, "w").write(text)
    if stdout:
        print(text)
    return None


def scatter_plot(x, y, xlabel, ylabel, out, 
                show_xy_axes=True,
                xerr=None, yerr=None,
                xlimits=None,
                ylimits=None,
                aspect="auto",

                same_xy_scale=True,
                show_regression_line=False,
                show_diagonal_line=False,

                show_rmse=False,
                show_armse=False,
                show_R=False,
                show_regression_line_eq=False,

                figure_size=(3.2, 3.2*6/8),
                dpi=300, 
                fontsize=8,
                font = {"fontname": "Arial"},
                markers=None,
                markersize=20, 
                markercolors=None,
                text_pos=[0.55, 0.1],
                line_styles = {"regression" : "k--", "diagonal" : "k-"},
                line_weights = {"regression" : 2, "diagonal" : 1} ):
    """
    """
    if show_diagonal_line:
        assert same_xy_scale, "to show diagonal line, x and y must be shown on the same scale"

    assert x.shape == y.shape, "x and y must have the same shape"
    if xerr is not None:
        assert xerr.shape == x.shape, "x and xerr must have the same shape"
        assert np.all(xerr >= 0), "xerr must be non-negative"

    if yerr is not None:
        assert yerr.shape == y.shape, "y and yerr must have the same shape"
        assert np.all(yerr >= 0), "yerr must be non-negative"

    if xerr is None:
        lower_x, upper_x = x.min(), x.max()
    else:
        lower_x = (x - xerr).min()
        upper_x = (x + xerr).max()

    if yerr is None:
        lower_y, upper_y = y.min(), y.max()
    else:
        lower_y = (y - yerr).min()
        upper_y = (y + yerr).max()

    if same_xy_scale:
        lower_x = np.min([lower_x, lower_y])
        lower_y = lower_x

        upper_x = np.max([upper_x, upper_y])
        upper_y = upper_x

    lower_x = np.floor(lower_x)
    upper_x = np.ceil(upper_x)
    lower_y = np.floor(lower_y)
    upper_y = np.ceil(upper_y)

    if xlimits is not None:
        lower_x, upper_x = xlimits

    if ylimits is not None:
        lower_y, upper_y = ylimits

    plt.figure(figsize=figure_size)
    if show_diagonal_line:
        plt.plot( [lower_x, upper_x], [lower_y, upper_y], line_styles["diagonal"], line_weights["diagonal"] )

    plt.axis(aspect=aspect)

    axes = plt.gca()
    axes.set_xlim([lower_x, upper_x])
    axes.set_ylim([lower_y, upper_y])


    if markercolors is None:
        markercolors = ["k" for i in range(len(x))]
    else:
        assert type(markercolors) == list, "markercolors must be either None or a list of str"
        assert len(markercolors) == len(x), "markercolors must have the same len as x and y"

    if markers is None:
        markers = ["o" for i in range(len(x))]
    else:
        assert type(markers) == list, "markers must be either None or a list of str"
        assert len(markers) == len(x), "markers must have the same len as x and y"

    if xerr is None and yerr is None:
        for i in range(len(x)):
            plt.errorbar(x[i], y[i], ms=markersize, marker=markers[i], c=markercolors[i], linestyle="None")

    if (xerr is not None) and (yerr is None):
        for i in range(len(x)):
            plt.errorbar(x[i], y[i], xerr=xerr[i], ms=markersize, marker=markers[i], c=markercolors[i], linestyle="None")

    if (xerr is None) and (yerr is not None):
        for i in range(len(x)):
            plt.errorbar(x[i], y[i], yerr=yerr[i], ms=markersize, marker=markers[i], c=markercolors[i], linestyle="None")

    if (xerr is not None) and (yerr is not None):
        for i in range(len(x)):
            plt.errorbar(x[i], y[i], xerr=xerr[i], yerr=yerr[i], ms=markersize, marker=markers[i], c=markercolors[i], linestyle="None")

    if show_regression_line:
        xl, yl, line_text = _regression_line(x, y)
        plt.plot(xl, yl, line_styles["regression"], lw=line_weights["regression"])

    text = ""
    if show_R:
        corr_coef, r_std = _correlation_coef(x, y)
        text += "Pearson's R = %0.2f"%(corr_coef)

    if show_rmse:
        mean_sq_error, rmse_std = _rmse(x,y)
        text += "\nRMSE = %0.2f"%(mean_sq_error)

    if show_armse:
        amean_sq_error, armse_std = _armse(x,y)
        text += "\naRMSE = %0.2f"%(amean_sq_error)

    if show_regression_line_eq:
        xl, yl, line_text = _regression_line(x, y)
        text += "\n%s"%line_text
    if len(text) > 0:
        plt.text( text_pos[0]*upper_x + ( 1-text_pos[0] )*lower_x, text_pos[1]*upper_y + ( 1-text_pos[1] )*lower_y, text, fontsize=fontsize, **font )

    ax = plt.axes()
    for tick in ax.xaxis.get_major_ticks():
        tick.label.set_fontsize(fontsize)

    for tick in ax.yaxis.get_major_ticks():
        tick.label.set_fontsize(fontsize)

    if not show_xy_axes:
        x_label = None
        y_label = None
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

    if xlabel is not None:
        plt.xlabel(xlabel, fontsize=fontsize, **font)
    if ylabel is not None:
        plt.ylabel(ylabel, fontsize=fontsize, **font)

    plt.tight_layout()
    plt.savefig(out, dpi=dpi)
    return None

# see http://matplotlib.org/api/markers_api.html
MARKERS = [".", ",", "o", "v", "^", "<", ">", "1", "2", "3", "4", "8", "s", "p", "*", "h", "H", "+", "x", "D", "d", "|", "_"]


def plot_lines(xy_list, xlabel, ylabel, out,
        figure_size=(3.2, 3.2*6/8),
        dpi=300,
        fontsize=8,
        font = {"fontname": "Arial"},
        lw=1.5,
        line_styles=None,
        markers=None,
        markersize=4,
        colors=None,
        limits=None,
        n_xtics=5):
    """
    """
    assert type(xy_list) == list, "xy_list must be a list of 2d arrays"
    if limits is None:
        x_min = np.min([xy[:,0].min() for xy in xy_list])
        x_max = np.max([xy[:,0].max() for xy in xy_list]) 
    else:
        x_min, x_max = limits[0], limits[1]
    xtics = np.linspace(x_min, x_max, n_xtics)
    xtic_labels = list(np.array(xtics, dtype=int))

    plt.figure(figsize=figure_size)
    if markers is None:
        markers = MARKERS

    if line_styles is None:
        line_styles = ["-" for i in range(len(xy_list))]

    # see http://matplotlib.org/examples/color/colormaps_reference.html   
    #if colors is None:
    #    cmap = matplotlib.cm.get_cmap('gnuplot')
    #    scalarMap = matplotlib.cm.ScalarMappable(norm=plt.normalize(min=0, max=1), cmap=cmap)
    #    colors = [ scalarMap.to_rgba(i) for i in np.linespace(0., 1., len(xy_list)) ]

    for i in range(len(xy_list)):
        if colors is None:
            plt.plot(xy_list[i][:,0], xy_list[i][:,1], 
                    linestyle=line_styles[i], 
                    marker=markers[i],
                    ms=markersize,
                    lw=lw)
        else:
            plt.plot(xy_list[i][:,0], xy_list[i][:,1],
                    linestyle=line_styles[i],
                    color=colors[i],
                    marker=markers[i],
                    ms=markersize,
                    lw=lw)

    ax = plt.axes()
    ax.set_xticks(xtics)
    #ax.set_xticklabels(xtic_labels)

    for tick in ax.xaxis.get_major_ticks():
        tick.label.set_fontsize(fontsize)
    
    for tick in ax.yaxis.get_major_ticks():
        tick.label.set_fontsize(fontsize)

    plt.xlabel(xlabel, fontsize=fontsize, **font)
    plt.ylabel(ylabel, fontsize=fontsize, **font)

    if limits is not None:
        axes = plt.gca()
        axes.set_xlim(limits[0:2])
        axes.set_ylim(limits[2:4])

    plt.tight_layout()
    plt.savefig(out, dpi=dpi)
    return None



def improved_plot_lines(xs, ys, xerrs=None, yerrs=None, xlabel=None, ylabel=None, out="out.pdf",
                        legends=None, legend_pos="best", legend_fontsize=8,
                        x_logscale=False,
                        y_logscale=False,
                        figure_size=(3.2, 3.2*6/8),
                        dpi=300,
                        tick_fontsize=8,
                        label_fontsize=8,
                        font = {"fontname": "Arial"},
                        lw=1.5,
                        line_styles=None,
                        markers=None,
                        markersize=4,
                        colors=None,
                        limits=None,
                        nticks=8):
    """
    """
    assert type(xs) == list, "xs must be a list of 1D array"
    assert type(ys) == list, "ys must be a list of 1D array"
    assert len(xs) == len(ys), "xs and ys must have the same len"
    if xerrs is not None:
        assert type(xerrs) == list and len(xerrs) == len(xs), "xerrs must be a list of same len as xs"
    if yerrs is not None:
        assert type(yerrs) == list and len(yerrs) == len(ys), "yerrs must be a list of same len as ys"

    if legends is not None:
        assert len(legends) == len(xs), "legends has wrong len"

    if limits is None:
        x_min = np.min([x.min() for x in xs])
        x_max = np.max([x.max() for x in xs])
    else:
        x_min, x_max = limits[0], limits[1]

    plt.figure(figsize=figure_size)
    if markers is None:
        markers = MARKERS

    if line_styles is None:
        line_styles = ["-" for i in range(len(xs))]

    for i in range(len(xs)):

        if colors is None:

            if (xerrs is None) and (yerrs is None):
                plt.plot(xs[i], ys[i], linestyle=line_styles[i], marker=markers[i], ms=markersize, lw=lw )

            elif (xerrs is not None) and (yerrs is None):
                plt.errorbar(xs[i], ys[i], xerr=xerrs[i], linestyle=line_styles[i], marker=markers[i], ms=markersize, lw=lw )

            elif (xerrs is None) and (yerrs is not None):
                plt.errorbar(xs[i], ys[i], yerr=yerrs[i], linestyle=line_styles[i], marker=markers[i], ms=markersize, lw=lw)

            elif (xerrs is not None) and (yerrs is not None):
                plt.errorbar(xs[i], ys[i], xerr=xerrs[i], yerr=yerrs[i], linestyle=line_styles[i], marker=markers[i], ms=markersize, lw=lw)

        else:

            if (xerrs is None) and (yerrs is None):
                plt.plot(xs[i], ys[i], linestyle=line_styles[i], color=colors[i], marker=markers[i], ms=markersize, lw=lw)

            elif (xerrs is not None) and (yerrs is None):
                plt.errorbar(xs[i], ys[i], xerr=xerrs[i], linestyle=line_styles[i], color=colors[i], marker=markers[i], ms=markersize, lw=lw)

            elif (xerrs is None) and (yerrs is not None):
                plt.errorbar(xs[i], ys[i], yerr=yerrs[i], linestyle=line_styles[i], color=colors[i], marker=markers[i], ms=markersize, lw=lw)

            elif (xerrs is not None) and (yerrs is not None):
                plt.errorbar(xs[i], ys[i], xerr=xerrs[i], yerr=yerrs[i], linestyle=line_styles[i], color=colors[i], marker=markers[i], ms=markersize, lw=lw)

    ax = plt.axes()
    ax.locator_params(axis='x', nbins=nticks)
    ax.locator_params(axis='y', nbins=nticks)

    for tick in ax.xaxis.get_major_ticks():
        tick.label.set_fontsize(tick_fontsize)

    for tick in ax.yaxis.get_major_ticks():
        tick.label.set_fontsize(tick_fontsize)

    if xlabel is not None:
        plt.xlabel(xlabel, fontsize=label_fontsize, **font)
    if ylabel is not None:
        plt.ylabel(ylabel, fontsize=label_fontsize, **font)

    if limits is not None:
        axes = plt.gca()
        axes.set_xlim(limits[0:2])
        axes.set_ylim(limits[2:4])

    if x_logscale:
        ax.set_xscale("log")
    if y_logscale:
        ax.set_yscale("log")

    if legends is not None:
        plt.legend(legends, loc=legend_pos, fancybox=False, fontsize=legend_fontsize)

    plt.tight_layout()
    plt.savefig(out, dpi=dpi)
    return None



#---------
# to deal with insets

def _sub_line_plot(      ax,
                    xs, ys, xerrs=None, yerrs=None, xlabel=None, ylabel=None,  
                    x_logscale=False, y_logscale=False,
                    xlimits=None, n_xtics=5, integer_xtics=False,
                    ylimits=None, n_ytics=5,

                    fontsize=8,
                    font = {"fontname": "Arial"},
                    lw=1.5,
                    line_styles=None,
                    markers=None,
                    markersize=4,
                    colors=None):
    """
    """
    assert type(xs) == list, "xs must be a list of 1D array"
    assert type(ys) == list, "ys must be a list of 1D array"
    assert len(xs) == len(ys), "xs and ys must have the same len"
    if xerrs is not None:
        assert type(xerrs) == list and len(xerrs) == len(xs), "xerrs must be a list of same len as xs"
    if yerrs is not None:
        assert type(yerrs) == list and len(yerrs) == len(ys), "yerrs must be a list of same len as ys"

    if xlimits is None:
        x_min = np.min([x.min() for x in xs])
        x_max = np.max([x.max() for x in xs])
    else:
        x_min, x_max = xlimits[0], xlimits[1]

    xtics = np.linspace(x_min, x_max, n_xtics)

    if integer_xtics:
        xtics = np.array(xtics, dtype=int)

    if ylimits is None:
        y_min = np.min([y.min() for y in ys])
        y_max = np.max([y.max() for y in ys])
    else:
        y_min, y_max = ylimits[0], ylimits[1]

    ytics = np.linspace(y_min, y_max, n_ytics)

    if markers is None:
        markers = MARKERS

    if line_styles is None:
        line_styles = ["-" for i in range(len(xs))]

    for i in range(len(xs)):

        if colors is None:

            if (xerrs is None) and (yerrs is None):
                ax.plot(xs[i], ys[i], linestyle=line_styles[i], marker=markers[i], ms=markersize, lw=lw)

            elif (xerrs is not None) and (yerrs is None):
                ax.errorbar(xs[i], ys[i], xerr=xerrs[i], linestyle=line_styles[i], marker=markers[i], ms=markersize, lw=lw)

            elif (xerrs is None) and (yerrs is not None):
                ax.errorbar(xs[i], ys[i], yerr=yerrs[i], linestyle=line_styles[i], marker=markers[i], ms=markersize, lw=lw)

            elif (xerrs is not None) and (yerrs is not None):
                ax.errorbar(xs[i], ys[i], xerr=xerrs[i], yerr=yerrs[i], linestyle=line_styles[i], marker=markers[i], ms=markersize, lw=lw)

        else:

            if (xerrs is None) and (yerrs is None):
                ax.plot(xs[i], ys[i], linestyle=line_styles[i], color=colors[i], marker=markers[i], ms=markersize, lw=lw)

            elif (xerrs is not None) and (yerrs is None):
                ax.errorbar(xs[i], ys[i], xerr=xerrs[i], linestyle=line_styles[i], color=colors[i], marker=markers[i], ms=markersize, lw=lw)

            elif (xerrs is None) and (yerrs is not None):
                ax.errorbar(xs[i], ys[i], yerr=yerrs[i], linestyle=line_styles[i], color=colors[i], marker=markers[i], ms=markersize, lw=lw)

            elif (xerrs is not None) and (yerrs is not None):
                ax.errorbar(xs[i], ys[i], xerr=xerrs[i], yerr=yerrs[i], linestyle=line_styles[i], color=colors[i], marker=markers[i], ms=markersize, lw=lw)

    ax.set_xticks(xtics)
    #ax.set_yticks(ytics)

    for tick in ax.xaxis.get_major_ticks():
        tick.label.set_fontsize(fontsize)

    for tick in ax.yaxis.get_major_ticks():
        tick.label.set_fontsize(fontsize)

    if xlabel is not None:
        ax.set_xlabel(xlabel, fontsize=fontsize, **font)
    if ylabel is not None:
        ax.set_ylabel(ylabel, fontsize=fontsize, **font)

    if xlimits is not None:
        ax.set_xlim(xlimits[0:2])
    if ylimits is not None:
        ax.set_ylim(ylimits[0:2])

    if x_logscale:
        ax.set_xscale("log")
    if y_logscale:
        ax.set_yscale("log")

    return ax


def main_inset_line_plot(   inset_pos, out,
                            xs, ys, inset_xs, inset_ys, 

                            xerrs=None, yerrs=None, xlabel=None, ylabel=None,
                            x_logscale=False, y_logscale=False, xlimits=None, n_xtics=5, integer_xtics=False, 
                            ylimits=None, n_ytics=5,

                            inset_xerrs=None, inset_yerrs=None, inset_xlabel=None, inset_ylabel=None, 
                            inset_x_logscale=False, inset_y_logscale=False, inset_xlimits=None, inset_n_xtics=5, inset_integer_xtics=False, 
                            inset_ylimits=None, inset_n_ytics=5,
                            
                            fontsize=8, lw=1.5, line_styles=None, markers=None, markersize=2, colors=None,
                            inset_fontsize=6, inset_lw=1, inset_line_styles=None, inset_markers=None, inset_markersize=1, inset_colors=None, 
                            font = {"fontname": "Arial"}, figure_size=(3.2, 3.2*6/8), dpi=300 ):
    """
    """
    fig, ax1 = plt.subplots(figsize=figure_size)
    # These are in unitless percentages of the figure size. (0,0 is bottom left)
    left, bottom, width, height = inset_pos
    ax2 = fig.add_axes([left, bottom, width, height])

    ax1 = _sub_line_plot( ax1, xs, ys, xerrs=xerrs, yerrs=yerrs, xlabel=xlabel, ylabel=ylabel, 
                            x_logscale=x_logscale, y_logscale=y_logscale, 
                            xlimits=xlimits, n_xtics=n_xtics, integer_xtics=integer_xtics, 
                            ylimits=ylimits, n_ytics=n_ytics,
                            fontsize=fontsize, font=font, lw=lw, line_styles=line_styles, markers=markers, markersize=markersize, colors=colors )

    ax2 = _sub_line_plot( ax2, inset_xs, inset_ys, xerrs=inset_xerrs, yerrs=inset_yerrs, xlabel=inset_xlabel, ylabel=inset_ylabel,
                            x_logscale=inset_x_logscale, y_logscale=inset_y_logscale,
                            xlimits=inset_xlimits, n_xtics=inset_n_xtics, integer_xtics=inset_integer_xtics,
                            ylimits=inset_ylimits, n_ytics=inset_n_ytics,
                            fontsize=inset_fontsize, font=font, lw=inset_lw, line_styles=inset_line_styles, markers=inset_markers, 
                            markersize=inset_markersize, colors=inset_colors )

    plt.tight_layout()
    plt.savefig(out, dpi=dpi)
    return None


