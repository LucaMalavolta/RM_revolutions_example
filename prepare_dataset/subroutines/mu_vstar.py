import numpy as np
import constants
from kepler_exo import kepler_true_anomaly_orbital_distance

""""
    must provide in input:
        array with BJD
        array with exposure time (same length as BJD)
        dictionary with stellar parameters
        dictionary with planetary parameters


star_dict = {}
star_dict['lambda'] = 0.000
star_dict['inclination'] = 90.0000
star_dict['vsini'] = 3.00
star_dict['alpha'] = 0.2300
star_dict['radius'] = 0.805
star_dict['ld_coefficients'] = [0.816, 0.00]
star_dict['ld_law'] = 'quadratic'

planet_dict = {}
planet_dict['P'] = 2.21857567
planet_dict['Tc'] = 2454279.436714
planet_dict['i'] = 85.710
planet_dict['e'] = 0.000
planet_dict['omega'] = 90.000
planet_dict['orbit'] = 'circular' # 'keplerian'
planet_dict['Rp_Rs'] = 0.15667
planet_dict['a_Rs'] = 8.863
planet_dict['reference_time'] = 00000

bjd = np.arange(2454279.3000, 2454279.600, 0.0035)
exptime = np.ones_like(bjd) * 300 # EXPTIME
"""


def compute_mu_vstar_grid(bjd, exptime, star_dict, planet_dict, input_ngrid=51, input_timestep=100):



    star_grid = {}   # write an empty dictionary
    star_grid['n_grid'] = input_ngrid #start filling the dictionary with relevant parameters
    star_grid['half_grid'] = int((star_grid['n_grid'] - 1) / 2)
    star_grid['time_step'] = input_timestep # in seconds



    if planet_dict['orbit'] == 'circular':
        # Tcent is assumed as reference time
        Tref = planet_dict['Tc']
        Tcent_Tref = 0.000
    else:
        Tref = planet_dict['reference_time']
        Tcent_Tref = planet_dict['Tc'] - Tref

    inclination_rad = planet_dict['i'] * constants.deg2rad
    omega_rad = planet_dict['omega'] * constants.deg2rad


    """ Coordinates of the centers of each grid cell (add offset) """
    star_grid['xx'] = np.linspace(-1.000000, 1.000000, star_grid['n_grid'], dtype=np.double)
    star_grid['xc'], star_grid['yc'] = np.meshgrid(star_grid['xx'], star_grid['xx'], indexing='xy')
    # check the Note section of the wiki page of meshgrid
    # https://docs.scipy.org/doc/numpy/reference/generated/numpy.meshgrid.html

    """ Distance of each grid cell from the center of the stellar disk """
    star_grid['rc'] = np.sqrt(star_grid['xc'] ** 2 + star_grid['yc'] ** 2)
    # Must avoid negative numbers inside the square root
    star_grid['inside'] = star_grid['rc'] < 1.000000
    # Must avoid negative numbers inside the square root
    star_grid['outside'] = star_grid['rc'] >= 1.000000


    """ Determine the mu angle for each grid cell, as a function of radius. """
    star_grid['mu'] = np.zeros([star_grid['n_grid'], star_grid['n_grid']],dtype=np.double)  # initialization of the matrix with the mu values
    star_grid['mu'][star_grid['inside']] = np.sqrt(1. - star_grid['rc'][star_grid['inside']] ** 2)

    """ Limb darkening law and coefficients """
    ld_coefficients = star_dict['ld_coefficients']
    if star_dict['ld_law'] == 'quadratic':
        star_grid['I'] = 1 - ld_coefficients[0]*(1. - star_grid['mu']) - ld_coefficients[1]*(1. - star_grid['mu'])**2
    else:
        print('ERROR: Selected limb darkening law not implemented')
        quit()

    star_grid['I'][star_grid['outside']] = 0.000

    """ Intensity normalization"""
    star_grid['I0'] = np.sum(star_grid['I'])
    star_grid['I'] /= star_grid['I0']


    """  2.2 Determine the Doppler shift to apply to the spectrum of each grid cell, from Cegla+2015 """

    star_grid['x_ortho'] = star_grid['xc'] * np.cos(star_dict['lambda'] * constants.deg2rad) \
        - star_grid['yc'] * np.sin(star_dict['lambda'] * constants.deg2rad)  # orthogonal distances from the spin-axis
    star_grid['y_ortho'] = star_grid['xc'] * np.sin(star_dict['lambda'] * constants.deg2rad) \
        + star_grid['yc'] * np.cos(star_dict['lambda'] * constants.deg2rad)

    star_grid['r_ortho'] = np.sqrt(star_grid['x_ortho'] ** 2 + star_grid['y_ortho'] ** 2)
    star_grid['z_ortho'] = np.zeros([star_grid['n_grid'], star_grid['n_grid']],
                                    dtype=np.double)  # initialization of the matrix
    star_grid['z_ortho'][star_grid['inside']] = np.sqrt(
        1. -star_grid['r_ortho'][star_grid['inside']] ** 2)

    """ rotate the coordinate system around the x_ortho axis by an agle: """
    star_grid['beta'] = (np.pi / 2.) - \
        star_dict['inclination'] * constants.deg2rad

    """ orthogonal distance from the stellar equator """
    ### Equation 7 in Cegla+2016
    star_grid['yp_ortho'] = star_grid['z_ortho'] * np.sin(star_grid['beta']) \
        + star_grid['y_ortho'] * np.cos(star_grid['beta'])

    ### Equation 6 in Cegla+2016
    star_grid['zp_ortho'] = star_grid['z_ortho'] * np.cos(star_grid['beta']) \
        + star_grid['y_ortho'] * np.sin(star_grid['beta'])


    """ stellar rotational velocity for a given position """
    # differential rotation is included considering a sun-like law
    star_grid['v_star'] = star_grid['x_ortho'] * star_dict['vsini'] * (
        1. -star_dict['alpha'] * star_grid['yp_ortho'] ** 2)
    # Null velocity for points outside the stellar surface
    star_grid['v_star'][star_grid['outside']] = 0.0

    """ working arrays for Eq. 1 and 9 of Cegla+2016"""
    star_grid['muI'] = star_grid['I'] * star_grid['mu']
    star_grid['v_starI'] = star_grid['I'] * star_grid['v_star']

    #import matplotlib.pyplot as plt
    #plt.imshow(star_grid['v_star'])
    #plt.show()

    eclipsed_flux = np.zeros_like(bjd)
    mean_mu = np.zeros_like(bjd)
    mean_vstar =  np.zeros_like(bjd)

    for i_obs, bjd_value in enumerate(bjd):
        n_oversampling = int(exptime[i_obs] / star_grid['time_step'])

        """recomputing the oversampling steps to homogeneously cover the
        full integration time """
        if n_oversampling % 2 == 0:
            n_oversampling += 1
            delta_step = exptime[i_obs] / n_oversampling / 86400.

        half_time = exptime[i_obs] / 2 / 86400.

        bjd_oversampling = np.linspace(bjd_value - half_time, bjd_value + half_time, n_oversampling, dtype=np.double)

        true_anomaly, orbital_distance_ratio = kepler_true_anomaly_orbital_distance(
            bjd_oversampling - Tref,
            Tcent_Tref,
            planet_dict['P'],
            planet_dict['e'],
            omega_rad,
            planet_dict['a_Rs'])

        """ planet position during its orbital motion, in unit of stellar radius"""
        # Following Murray+Correia 2011 , with the argument of the ascending node set to zero.
        # 1) the ascending node coincide with the X axis
        # 2) the reference plance coincide with the plane of the sky
        planet_position = {
            'xp': -orbital_distance_ratio * (np.cos(omega_rad + true_anomaly)),
            'yp': orbital_distance_ratio * (np.sin(omega_rad + true_anomaly) * np.cos(inclination_rad)),
            'zp': orbital_distance_ratio * (np.sin(inclination_rad) * np.sin(omega_rad + true_anomaly))
        }
        # projected distance of the planet's center to the stellar center
        planet_position['rp'] = np.sqrt(planet_position['xp']**2  + planet_position['yp']**2)

        # iterating on the sub-exposures
        I_sum = 0.00
        muI_sum = 0.00
        vstarI_sum = 0.00

        for j, zeta in enumerate(planet_position['zp']):

            if zeta > 0 and planet_position['rp'][j] < 1. + planet_dict['Rp_Rs']:
                # the planet is in the foreground or inside the stellar disk, continue
                # adjustment: computation is performed even if only part of the planet is shadowing the star

                rd = np.sqrt((planet_position['xp'][j] - star_grid['xc']) ** 2 +
                                (planet_position['yp'][j] - star_grid['yc']) ** 2)

                """ Seelction of the portion of stars covered by the planet"""
                sel_eclipsed = (rd <= planet_dict['Rp_Rs']) & star_grid['inside']

                I_sum +=  np.sum(star_grid['I'][sel_eclipsed])
                muI_sum +=  np.sum(star_grid['muI'][sel_eclipsed])
                vstarI_sum += np.sum(star_grid['v_starI'][sel_eclipsed])

        if muI_sum > 0:
            eclipsed_flux[i_obs] = I_sum/n_oversampling
            mean_mu[i_obs] = muI_sum/I_sum
            mean_vstar[i_obs] = vstarI_sum/I_sum


    return mean_mu, mean_vstar, eclipsed_flux, star_grid

"""
mean_mu, mean_vstar, eclipsed_flux, star_grid = \
    compute_mu_vstar_grid(bjd, exptime, star_dict, planet_dict, input_ngrid=1001)

import matplotlib.pyplot as plt
plt.imshow(star_grid['v_star'], origin='lower', cmap='seismic')
plt.show()
plt.plot(bjd,mean_mu)
plt.scatter(bjd,mean_mu, s=5, c='C1')
plt.xlabel('BJD')
plt.ylabel('<mu>')
plt.show()
plt.plot(bjd,mean_vstar)
plt.scatter(bjd,mean_vstar, s=5, c='C1')
plt.xlabel('BJD')
plt.ylabel('vstar')
plt.show()
plt.plot(mean_mu,mean_vstar)
plt.xlabel('mean_mu')
plt.ylabel('vstar')
plt.show()

plt.plot(bjd,1.-eclipsed_flux)
plt.show()

fileout = open('simulated_data.dat', 'w')
fileout.write('#epoch value error jitter offset subset exptime\n')
for b, m in zip(bjd, mean_vstar):
    fileout.write('{0:16f} {1:16f} 0.20 -1 -1 -1 300 \n'.format(b,m))
fileout.close()
"""