from __future__ import print_function, division
import numpy as np
from scipy.interpolate import interp1d

__all__ = ['rebin_ccf']

def rebin_exact_flux(wave_in, step_in, flux_in, wave_out, step_out,
                     quadrature=False,
                     preserve_flux=True):
    """
    Previously named rebin_order
    :param wave_in:
    :param step_in:
    :param flux_in:
    :param wave_out:
    :param step_out:
    :param quadrature:
    :param preserve_flux:
    :return:

    Spectral rebinning with flux conservation
    """
    if quadrature:
        flux_in = flux_in**2.

    flux_out = np.zeros(np.shape(wave_out), dtype=np.double)
    n1 = np.size(wave_in)
    n2 = np.size(wave_out)
    ns_prv = 0

    for i in range(0, n2):
        # print i, ' of ', n2
        # Starting and ending point of the bin
        wlb = wave_out[i] - step_out[i] / 2.000
        wle = wave_out[i] + step_out[i] / 2.000

        # Normalized flux value within the bin
        fl_nm = 0.00

        # b->blue and r->red side of the original spectrum which include the bin
        # ib and ie are initialized with values close to the ones of the last iteration to save time
        ib = ns_prv
        ir = ns_prv

        for ns in range(ns_prv, n1 - 1):
            # simple algorithm to search the closest indexes near the bin boundaries
            if wave_in[ib] + step_in[ib] / 2.00 < wlb: ib += 1
            if wave_in[ir] + step_in[ir] / 2.00 < wle: ir += 1

            # when we are close to the boundary of the spectra, we stop
            if ir < ns - 3: break

        # Fail-safe checks
        if ib > ns_prv: ns_prv = ib - 3
        if ib < 0 or ir > n1: continue
        if ib > ir: continue
        if ns_prv < 0: ns_prv = 0

        # Now the true rebinning section
        if ib == ir:
            pix_s = (wle - wlb) / step_in[ib]  # fraction
            pix_e = 0.
            flux_out[i] += pix_s * flux_in[ib]
            fl_nm += pix_s
        elif ib + 1 == ir:
            pix_s = (wave_in[ib] + step_in[ib] * 0.5 - wlb) / step_in[ib]
            pix_e = (wle - (wave_in[ir] - step_in[ir] * 0.5)) / step_in[ir]
            flux_out[i] += (pix_s * flux_in[ib] + pix_e * flux_in[ir])
            fl_nm += (pix_s + pix_e)
        else:
            pix_s = (wave_in[ib] + step_in[ib] * 0.5 - wlb) / step_in[ib]
            pix_e = (wle - (wave_in[ir] - step_in[ir] * 0.5)) / step_in[ir]
            flux_out[i] += (pix_s * flux_in[ib] + pix_e * flux_in[ir])
            fl_nm += (pix_s + pix_e)
            for j in range(ib + 1, ir):
                flux_out[i] += flux_in[j]
                fl_nm += 1.00
        if (not preserve_flux) and fl_nm > 0.0:
            if quadrature:
                fl_nm *= fl_nm
            flux_out[i] /= fl_nm

    if quadrature:
        return np.sqrt(flux_out)
    else:
        return flux_out

def rebin_with_interpolation(wave_in, step_in, flux_in, wave_out, step_out,
                             quadrature=False,
                             preserve_flux=True,
                             interp_kind='cubic'):

    ndata = len(wave_in)

    normalization_factor = 1.0
    if preserve_flux:
        step_in_internal = np.ones(ndata)
        step_out_internal = np.ones(len(step_out))
    else:
        step_in_internal = step_in
        step_out_internal = step_out
        if quadrature:
            normalization_factor = (np.median(step_out) / np.median(step_in))
    if quadrature:
        flux_in = np.power(flux_in, 2.)

    wave_in_cumul = np.zeros(ndata+1)
    flux_in_cumul = np.zeros(ndata+1)

    flux_in_cumul[0] = 0.0

    wave_in_cumul[0] = wave_in[0] - step_in[0] / 2.0

    for i in range(1, ndata):
        flux_in_cumul[i] = flux_in_cumul[i - 1] + flux_in[i - 1] * step_in_internal[i - 1]
        # wave_in_cumul[i] = wave_in[i]-step_in[i]/2.
        wave_in_cumul[i] = wave_in[i] - (wave_in[i] - wave_in[i - 1]) / 2.

    flux_in_cumul[ndata] = flux_in_cumul[ndata - 1] + flux_in[ndata - 1] * step_in_internal[ndata - 1]
    wave_in_cumul[ndata] = wave_in[ndata - 1] + step_in[ndata - 1] / 2.

    flux_cumul_interp1d = interp1d(wave_in_cumul, flux_in_cumul, kind=interp_kind, bounds_error=False, fill_value=0.000)

    flux_out = (flux_cumul_interp1d(wave_out + step_out / 2.) - flux_cumul_interp1d(
        wave_out - step_out / 2.)) / step_out_internal

    if quadrature:
        return np.sqrt(flux_out) / normalization_factor
    else:
        return flux_out


def rebin_ccf(range_in, step_in, ccf_in, range_out, step_out,
                   rv_shift=None,
                   is_error=False,
                   quadrature=False,
                   preserve_flux=True,
                   method='cubic_interpolation'):

    if is_error:
        quadrature = True
        method='exact_flux'

    if rv_shift:
        range_in = range_in+rv_shift

    if method == 'exact_flux':
        ccf_out = rebin_exact_flux(range_in, step_in, ccf_in, range_out, step_out,
                                    quadrature=quadrature, preserve_flux=preserve_flux)
    elif method == 'cubic_interpolation':
        ccf_out = rebin_with_interpolation(range_in, step_in, ccf_in, range_out, step_out,
                                            quadrature=quadrature, preserve_flux=preserve_flux, interp_kind='cubic')


    else:
        raise ValueError("method ", method, 'not supported by rebinning subroutine')

    return ccf_out
