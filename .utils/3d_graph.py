import sys
import os
import math
import numpy as np
from scipy import integrate
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as plticker
from mpl_toolkits.mplot3d import axes3d

cwd = os.getcwd()
sys.path.insert(0, cwd + '/AntineutrinoSpectrum')

import latex
from spectrum import OscillatedSpectrum


def gaussian(_x, _sigma):
    _const = math.sqrt(2 * math.pi)
    _appo = 1 / _const / _sigma * np.exp(- np.power(_x, 2) / 2. / (_sigma ** 2))
    return _appo


E = np.arange(1.806, 10.01, 0.01)  # in MeV
Edep = E - 0.8

spectrum = OscillatedSpectrum(t12=0.310, m21=7.39e-5, t13_N=0.02240, m3l_N=2.525e-3,
                              t13_I=0.02263, m3l_I=-2.512e-3)

spectrum.osc_spectrum(E, 0, plot_this=False)
zero = np.zeros(len(E))

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot(E, zero, spectrum.norm_osc_spect_N, 'b', linewidth=1., label='NO')

a = 0.029
b = 0.008

Gs = np.empty((len(E), len(E)))
Gs_norm = np.empty((len(E), len(E)))
n_ = 0
for E0 in Edep:
    appo = math.pow(a, 2) / E0 + math.pow(b, 2)
    sigma = math.sqrt(appo) * E0
    g = gaussian(Edep - E0, sigma)
    # Gs[n_, :] = g
    Gs[:, n_] = g
    g_appo = g * spectrum.norm_osc_spect_N[n_]
    Gs_norm[n_, :] = g_appo
    n_ += 1

g_sum = np.empty(len(E))
g_sum_simps = np.empty(len(E))
for n0 in np.arange(0, len(E)):
    g_sum[n0] = Gs_norm[:, n0].sum()
    g_sum_simps[n0] = integrate.simps(Gs_norm[:, n0], E)

ax.plot(zero, Edep, g_sum_simps, 'r', linewidth=1., label='convolution')

n = np.array([20, 70, 120, 170, 220, 270, 320, 370, 420, 470, 520, 570, 620, 670, 720])  # al MeV e al mezzo MeV
for n0 in n:
    x = E[n0]  # fixed Enu
    xvis = x - 0.8
    appo = math.pow(a, 2) / xvis + math.pow(b, 2)
    sigma = math.sqrt(appo) * xvis
    # sigma = 1.
    g = gaussian(Edep - xvis, sigma)
    g_appo = g * spectrum.norm_osc_spect_N[n0]
    x_appo = np.empty(len(E))
    x_appo.fill(x)
    ax.plot(x_appo, Edep, g_appo, linewidth=1., label=r'sigma = %.4f $10^{-2}$' % (sigma * 100))

### 3D plot of a graphical representation of the numerical convolution
fig.suptitle('Convolution via a graphic method')
ax.set_xlabel(r'$\text{E}_{\nu}$ [\si{MeV}]')
ax.set_xlim(0., 11.)
ax.set_ylabel(r'$\text{E}_{\text{vis}}$ [\si{MeV}]')
ax.set_ylim(0., 11.)
ax.set_zlabel(r'N [arb. unit]')
# ax.set_zlim(-0.005,0.095)
ax.legend(prop={'size': 8})
fig.savefig('SpectrumPlots/3d_convolution.pdf', format='pdf')
print('\nThe plot has been saved in SpectrumPlots/3d_convolution.pdf')

### plot of the matrix of the detector's response
loc = plticker.MultipleLocator(base=2.0)
loc1 = plticker.MultipleLocator(base=0.5)

fig2 = plt.figure()
ax2 = fig2.add_subplot(111)
fig2.subplots_adjust(left=0.02, right=0.95, top=0.95)
im = ax2.imshow(Gs, cmap='Blues', norm=mpl.colors.Normalize(), interpolation='none', origin={'lower', 'lower'},
                extent=[Edep[0], Edep[-1], Edep[0], Edep[-1]], vmin=abs(Gs).min())
bar = fig2.colorbar(im)
ax2.set_xlabel(r'$E_{\text{dep}}$ [\si{MeV}]')
# ax2.set_xlim(1., 10.)
ax2.set_ylabel(r'$E_{\text{vis}}$ [\si{MeV}]')
bar.set_label(r'$G(E_{\text{vis}} - E_{\text{dep}}, \delta E_{\text{dep}})$ [arb. unit] ')
ax2.xaxis.set_major_locator(loc)
ax2.xaxis.set_minor_locator(loc1)
ax2.yaxis.set_major_locator(loc)
ax2.yaxis.set_minor_locator(loc1)
ax2.tick_params('both', direction='out', which='both')
# ax2.set_title(r'Detector response')
fig2.savefig('SpectrumPlots/response.pdf', format='pdf')
print('\nThe plot has been saved in SpectrumPlots/response.pdf')

# ### plot of the result of the convolution
# fig1 = plt.figure()
# fig1.suptitle(r'3d convolution')
# ax1 = fig1.add_subplot(111)
# # ax1.plot(Evis,spectrum.norm_osc_spect_N,'r',linewidth=1.,label='NO')
# ax1.plot(Evis, g_sum_simps, 'b', linewidth=1., label='convolution - simps')
# ax1.set_xlabel(r'$\text{E}_{\text{vis}}$ [\si{MeV}]')
# ax1.set_ylabel(r'N [arb. unit]')
# ax1.set_ylim(-0.005, 0.095)
# ax1.grid()
# ax1.legend()
# fig1.savefig('SpectrumPlots/conv_3d.pdf', format='pdf')
# print('\nThe plot has been saved in SpectrumPlots/conv_3d.pdf')

plt.ion()
plt.show()
