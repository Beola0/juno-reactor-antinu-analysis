import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

unfolded_spectrum = pd.read_csv("total_unfolded_DYB.txt", sep="\t",
                                names=["bin_center", "IBD_spectrum", "reactor_flux"], header=0)
unfolded_u235 = pd.read_csv("u235_unfolded_DYB.txt", sep="\t",
                            names=["bin_center", "IBD_spectrum", "reactor_flux"], header=0)
unfolded_pu_combo = pd.read_csv("pu_combo_unfolded_DYB.txt", sep="\t",
                                names=["bin_center", "IBD_spectrum", "reactor_flux"], header=0)
sum_ = unfolded_u235["IBD_spectrum"] * 0.564 + unfolded_pu_combo["IBD_spectrum"] * 0.304

plt.figure()
plt.plot(unfolded_spectrum["bin_center"], unfolded_spectrum["IBD_spectrum"], "b", label='total')
plt.plot(unfolded_u235["bin_center"], unfolded_u235["IBD_spectrum"], "r--", label='u235')
plt.plot(unfolded_pu_combo["bin_center"], unfolded_pu_combo["IBD_spectrum"], "g-.", label='pu_combo')
plt.plot(unfolded_pu_combo["bin_center"], sum_, "k", label='u235 + pu_combo')
plt.xlabel("bin center [MeV]")
plt.ylabel("IBD_spectrum [cm^2/fission/MeV]")
plt.grid()
plt.legend()

plt.figure()
plt.plot(unfolded_spectrum["bin_center"], unfolded_spectrum["reactor_flux"], "b", label='total')
plt.plot(unfolded_u235["bin_center"], unfolded_u235["reactor_flux"], "r--", label='u235')
plt.plot(unfolded_pu_combo["bin_center"], unfolded_pu_combo["reactor_flux"], "g-.", label='pu_combo')
plt.xlabel("bin center [MeV]")
plt.ylabel("reactor_flux [cm^2/fission/MeV]")
plt.grid()
plt.legend()

plt.ion()
plt.show()
