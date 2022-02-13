import numpy as np
from scipy import interpolate
import matplotlib.pyplot as plt

import os
import sys
cwd = os.getcwd()
sys.path.insert(0, cwd + '/AntineutrinoSpectrum')

path = '/Users/beatricejelmini/Desktop/JUNO/JUNO_codes/JUNO_ReactorNeutrinosAnalysis/Inputs/spectra'

covariance_dyb = np.loadtxt(path+'/CovMatrix_unfolding_DYB.txt')
N = 75
corr_dyb = np.zeros((N, N))
for i in np.arange(0,N,1):
    for j in np.arange(0,N,1):
        corr_dyb[i,j] = covariance_dyb[i,j]/np.sqrt(covariance_dyb[i,i])/np.sqrt(covariance_dyb[j,j])

full_covariance_dyb_pp = np.loadtxt(path+'/CovMatrix_unfolding_DYB_PROSPECT.txt')
full_covariance_dyb_pp=full_covariance_dyb_pp*1.e-86
N = 50
corr_dyb_pp = np.zeros((N, N))
for i in np.arange(0,N,1):
    for j in np.arange(0,N,1):
        corr_dyb_pp[i,j] = full_covariance_dyb_pp[i,j]/np.sqrt(full_covariance_dyb_pp[i,i])/np.sqrt(full_covariance_dyb_pp[j,j])



plt.figure()
plt.imshow(full_covariance_dyb_pp, origin='lower')
plt.colorbar()

plt.figure()
plt.imshow(corr_dyb_pp, origin='lower', vmin=-1, vmax=1)
plt.colorbar()

plt.figure()
plt.imshow(covariance_dyb, origin='lower')
plt.colorbar()

plt.figure()
plt.imshow(corr_dyb, origin='lower', vmin=-1, vmax=1)
plt.colorbar()

plt.ion()
plt.show()
