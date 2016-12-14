import matplotlib as mpl
mpl.use('Agg')

import matplotlib.pyplot as plt

plt.style.use('ggplot')
mpl.rcParams['text.color'] = 'black'
mpl.rcParams['lines.linewidth'] = 2
mpl.rcParams['font.family'] = 'sans-serif'
mpl.rcParams['font.size'] = 8
mpl.rcParams['axes.facecolor'] = 'white'
mpl.rcParams['axes.edgecolor'] = 'black'
mpl.rcParams['axes.labelcolor'] = 'black'
mpl.rcParams['axes.grid'] = True
# mpl.rcParams['axes.xmargin'] = 0.001
# mpl.rcParams['axes.ymargin'] = 0.01
mpl.rcParams['grid.color'] = 'black'
mpl.rcParams['grid.linestyle'] = '-'
mpl.rcParams['grid.alpha'] = 0.6
mpl.rcParams['xtick.color'] = 'black'
mpl.rcParams['ytick.color'] = 'black'
mpl.rcParams['figure.figsize'] = (5, 4)
# mpl.rcParams['figure.dpi'] = 300
mpl.rcParams['savefig.dpi'] = 300
mpl.rcParams['savefig.format'] = 'png'

