import numpy as np

sin2_t12 = 0.310
s12 = np.sqrt(sin2_t12)
c12 = np.sqrt(1. - sin2_t12)

# Normal ordering

sin2_t13 = 2.24e-2
sin2_t23 = 0.582

s13 = np.sqrt(sin2_t13)
s23 = np.sqrt(sin2_t23)

c13 = np.sqrt(1. - sin2_t13)
c23 = np.sqrt(1. - sin2_t23)

Ue1_N = c12 * c13
Ue2_N = s12 * c13
Ue3_N = s13

Umu1_N = - s12 * c23 - c12 * s23 * s13
Umu2_N = c12 * c23 - s12 * s23 * s13
Umu3_N = c13 * s23

Utau1_N = s12 * s23 - c12 * s13 * c23
Utau2_N = - c12 * s23 - s12 * s13 * c23
Utau3_N = c13 * c23

sum_e_N = Ue1_N**2 + Ue2_N**2 + Ue3_N**2
sum_mu_N = Umu1_N**2 + Umu2_N**2 + Umu3_N**2
sum_tau_N = Utau1_N**2 + Utau2_N**2 + Utau3_N**2

sum_1_N = Ue1_N**2 + Umu1_N**2 + Utau1_N**2
sum_2_N = Ue2_N**2 + Umu2_N**2 + Utau2_N**2
sum_3_N = Ue3_N**2 + Umu3_N**2 + Utau3_N**2

# Inverted ordering

sin2_t13 = 2.263e-2
sin2_t23 = 0.582

s13 = np.sqrt(sin2_t13)
s23 = np.sqrt(sin2_t23)

c13 = np.sqrt(1. - sin2_t13)
c23 = np.sqrt(1. - sin2_t23)

Ue1_I = c12 * c13
Ue2_I = s12 * c13
Ue3_I = s13

Umu1_I = - s12 * c23 - c12 * s23 * s13
Umu2_I = c12 * c23 - s12 * s23 * s13
Umu3_I = c13 * s23

Utau1_I = s12 * s23 - c12 * s13 * c23
Utau2_I = - c12 * s23 - s12 * s13 * c23
Utau3_I = c13 * c23

sum_e_I = Ue1_I**2 + Ue2_I**2 + Ue3_I**2
sum_mu_I = Umu1_I**2 + Umu2_I**2 + Umu3_I**2
sum_tau_I = Utau1_I**2 + Utau2_I**2 + Utau3_I**2

sum_1_I = Ue1_I**2 + Umu1_I**2 + Utau1_I**2
sum_2_I = Ue2_I**2 + Umu2_I**2 + Utau2_I**2
sum_3_I = Ue3_I**2 + Umu3_I**2 + Utau3_I**2
