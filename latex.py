import matplotlib.pyplot as plt

## Note: packages like siunitx or mhchem are no longer working
pgf_with_latex = {
    "pgf.texsystem": "pdflatex",
    "text.usetex": True,
    "font.family": "serif",
    "font.sans-serif": ["Computer Modern Roman"],
    "axes.labelsize": 16,               # LaTeX default is 10pt font.
    "font.size": 16,
    "legend.fontsize": 13,               # Make the legend/label fonts
    "xtick.labelsize": 13,               # a little smaller
    "ytick.labelsize": 13,
    "pgf.preamble": [
        r"\usepackage[utf8x]{inputenc}",    # use utf8 fonts
        r"\usepackage[T1]{fontenc}",        # plots will be generated
        r"\usepackage{amsmath,amsfonts,amssymb}"
        # r"\usepackage{mhchem}"
        ]                                   # using this preamble
}

# mpl.use("pgf")  # does not work with plt.show()
plt.rcParams.update(pgf_with_latex)
