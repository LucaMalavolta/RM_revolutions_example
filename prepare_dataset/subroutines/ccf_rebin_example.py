import numpy as np
import matplotlib.pyplot as plt 

rv_zero = 7.4354
sigma = 0.15454

step = 0.25
n_step = 90

ccf_x0 = np.arange(-step*n_step, (n_step+1)*step, step)
ccf_s0 = np.ones_like(ccf_x0) * step
ccf_y0 = 1 - np.exp(-(ccf_x0-rv_zero)**2/(2*sigma**2))

from rebin import rebin_ccf

## New range for rebinning, with different step to highlight the differences
#between flux conservation or flux interpolation

new_step = 0.35
new_n_step = 40

ccf_x1 = np.arange(-new_step*new_n_step, (new_n_step+1)*new_step, new_step)
ccf_s1 = np.ones_like(ccf_x1) * new_step

ccf_y1_interpolation = rebin_ccf(ccf_x0, ccf_s0, ccf_y0, ccf_x1, ccf_s1,
                   rv_shift=-rv_zero, preserve_flux=False)

ccf_y1_flux_conservation = rebin_ccf(ccf_x0, ccf_s0, ccf_y0, ccf_x1, ccf_s1,
                   rv_shift=-rv_zero, preserve_flux=True)

plt.plot(ccf_x0, ccf_y0, c='C0', label='Input')
plt.plot(ccf_x0-rv_zero, ccf_y0, c='C1', label='Shifted input')


plt.plot(ccf_x1, ccf_y1_interpolation, c='C2', label='Interpolation')
plt.plot(ccf_x1, ccf_y1_flux_conservation, c='C3', label='Flux conservation')
plt.legend()
plt.show()
