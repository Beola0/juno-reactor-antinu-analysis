import matplotlib as mpl
# from matplotlib import rc

# plt.rc('text',usetex=True)
mpl.rcParams['text.latex.preamble'] = ['\\usepackage{siunitx}']
# mpl.rcParams['text.latex.preamble'] = ['\\usepackage{mhchem}']
pgf_with_latex = {  # setup matplotlib to use latex for output# {{{
    "pgf.texsystem": "pdflatex",  # change this if using xetex or lautex
    "text.usetex": True,  # use LaTeX to write all text
    "font.family": "serif",
    "font.serif": ['Computer Modern'],  # blank entries should cause plots
    "font.sans-serif": [],  # to inherit fonts from the document
    "font.monospace": [],
    "axes.labelsize": 16,  # LaTeX default is 10pt font.
    "font.size": 16,
    "legend.fontsize": 16,  # Make the legend/label fonts
    "xtick.labelsize": 13,  # a little smaller
    "ytick.labelsize": 13,
    #    "figure.figsize": figsize(0.9),     # default fig size of 0.9 textwidth
    "pgf.preamble": [
        r"\usepackage[utf8x]{inputenc}",  # use utf8 fonts
        r"\usepackage[T1]{fontenc}",  # plots will be generated
        r"\usepackage[detect-all,locale=DE]{siunitx}",
    ]  # using this preamble
}
# }}}
mpl.rcParams.update(pgf_with_latex)
