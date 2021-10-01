import os
import sys
import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as plticker

cwd = os.getcwd()
sys.path.insert(0, cwd)

import latex


a = 0.029
b = 0.008

E_nu = np.arange(1.806, 10.01, 0.001)  # in MeV
Edep = E_nu - 0.8

appo = math.pow(a, 2) / Edep + math.pow(b, 2)
sigma_Edep = np.sqrt(appo) * Edep

# a2 = 0.03
# b2 = 0.
# sigma_E = a2 * np.sqrt(Edep)

fig = plt.figure()
ax = fig.add_subplot(111)

loc = plticker.MultipleLocator(base=2.0)
loc1 = plticker.MultipleLocator(base=0.5)

ax.set_xlabel(r'$E_{\text{dep}}$ [\si{MeV}]')
ax.set_xlim(0., 10.)
ax.set_ylabel(r'$\sigma_E$ [\si{MeV}]', color='b')
ax.set_ylim(0.02, 0.12)
ax.plot(Edep, sigma_Edep, 'b', linewidth=1)
# ax.plot(Edep, sigma_E, 'g', linewidth=1)
ax.tick_params(axis='y', labelcolor='b')

ax.xaxis.set_major_locator(loc)
ax.xaxis.set_minor_locator(loc1)
ax.tick_params('both', direction='out', which='both')

# ax.text(4.01, 0.105, r'a = \SI{%.1f}{\percent}' % (a * 100) + '\nb = \SI{%.1f}{\percent}' % (b * 100))
# ax.set_title('Energy resolution')
ax.grid(alpha=0.45)

ax1 = ax.twinx()  # new axes with same x
ax1.set_ylabel(r'$\sigma_E/E~(\si{\percent})$', color='r')
ax1.set_ylim(1.15, 3.15)
ax1.plot(Edep, sigma_Edep / Edep * 100, 'r--', linewidth=1)
ax1.tick_params(axis='y', labelcolor='r')
# ax1.plot(Edep, sigma_E / Edep * 100, 'g-.', linewidth=1)

fig.tight_layout()

fig.subplots_adjust(left=0.13, right=0.88, top=0.95, bottom=0.12)

fig.savefig('SpectrumPlots/resolution.pdf', format='pdf', transparent=True)
print('\nThe plot has been saved in SpectrumPlots/resolution.pdf')

plt.ion()
plt.show()
