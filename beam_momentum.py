"""
M11 experiment steps with a data run that had actual beam data.

Things this does:
- makes a histogram of ToF data where we triggered on BC3/2 coincidences
- fits Gaussians for electrons, muons, and pions
- calculates the offset time due to cable length etc
- eventually calculates beam momentum by comparing to calculated values
"""
from argparse import ArgumentParser
import uproot
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from scipy.optimize import curve_fit

# put your file name here (can override this with -f argument)
DEFAULT_ROOT_FILE_NAME = 'output_000496.root'

# a couple helper functions for curve fitting:


def gauss(xvals, amp, mean, var):
    """Return a Gaussian with the given x values and parameters.

    xvals: numpy array of x values
    amp: number, amplitude of Gaussian
    mean: number, mean of Gaussian
    var: variance of Gaussian
    """
    return amp*np.exp(-(xvals-mean)**2/(2*var))


def comb_gauss(xvals, *params):
    """Return a sum of Gaussians.

    xvals: numpy array of x values
    params: list of parameters to pass, must have length being a multiple of 3
            (amplitude, mean, variance) for each Gaussian
    """
    assert len(params) % 3 == 0
    distr = 0
    for j in range(len(params) // 3):
        distr += gauss(xvals, *[params[j*3], params[j*3+1], params[j*3+2]])
    return distr


# the main function to calculate beam momentum:


def calculate_momentum(root_file_path):
    """given a root file, calculate beam momentum"""
    # constants
    length = 3.18  # measured length in meters, between the two scintillators
    c_m_per_ns = 0.2997925  # speed of light in m/ns

    # set plot parameters
    matplotlib.rcParams.update({'font.size': 22})

    # open root file, put adc/tdc data in numpy arrays
    print("opening file, reorganizing data")
    beam_file = uproot.open(root_file_path)
    beam_data = beam_file['midas_data']
    tdc = beam_data['tdc_value'].array(library="np", entry_start=0)
    adc = beam_data['adc_value'].array(library="np", entry_start=0)

    # put adc and tdc data into dicts with keys corresponding to detectors
    # BC1/2/3 = Scintillator number 1/2/3
    # C2/3/4/5/6 = PMT number 2/3/4/5/6
    adc_data = {'BC1': [], 'BC2': [], 'BC3': [],
                'C2': [], 'C3': [], 'C4': [], 'C5': [], 'C6': []}
    for adc_i in adc:
        adc_data['BC1'].append(adc_i[0])
        adc_data['BC2'].append(adc_i[1])
        adc_data['BC3'].append(adc_i[2])
        adc_data['C2'].append(adc_i[3])
        adc_data['C3'].append(adc_i[4])
        adc_data['C4'].append(adc_i[5])
        adc_data['C5'].append(adc_i[6])
        adc_data['C6'].append(adc_i[7])
    tdc_data = {'BC1': [], 'BC2': [], 'BC3': []}
    for tdc_i in tdc:
        tdc_data['BC1'].append(tdc_i[1])
        tdc_data['BC2'].append(tdc_i[2])
        tdc_data['BC3'].append(tdc_i[3])

    # cast as arrays so it's faster
    for k in adc_data:
        adc_data[k] = np.array(adc_data[k])
    for k in tdc_data:
        tdc_data[k] = np.array(tdc_data[k])

    print("fitting curves, making histogram")

    # calculate TOF delta t using BC3-BC1 (divide by 10 so it's in nanoseconds)
    delta_t = (tdc_data['BC3']-tdc_data['BC1']) / 10

    # mask zeros so we don't get spikes
    mask = delta_t != 0

    # make a histogram
    plt.figure(figsize=(12, 8))
    plt.title('beam')
    plt.xlabel('time-of-flight: BC3-BC1, ns')
    plt.ylabel('counts')
    hist, bins, _ = plt.hist(
        delta_t[mask], 179, range=(-10, 8), histtype='step', lw=3)

    # fit a curve
    bin_centres = bins[:-1] + (bins[1]-bins[0])/2
    coeff, *_ = curve_fit(
        # p0 = initial guesses, sorry about the magic numbers here
        comb_gauss, bin_centres, hist,
        p0=[4000, -5, 1, 16000, -1, 1, 4000, 2, 1])
    fit_hist = comb_gauss(bin_centres, *coeff)

    # get a mask for just the electrons, muons, pions
    e_sigma = np.sqrt(coeff[2])
    e_mean = coeff[1]
    m_mean = coeff[4]
    p_mean = coeff[7]

    mask_e = mask*(delta_t > e_mean-2*e_sigma)*(delta_t < e_mean+2*e_sigma)

    # plot the fit histogram
    plt.plot(bin_centres, fit_hist, c='k', lw=3, ls='--')

    # plot individual Gaussians
    plt.plot(bin_centres, gauss(bin_centres, *coeff[0:3]),
             c='r', lw=3, ls=':',
             label=f'$e$: mean = {coeff[1]:.2f}')
    plt.plot(bin_centres, gauss(bin_centres, *coeff[3:6]),
             c='orange', lw=3, ls=':',
             label=f'$\\mu$: mean = {coeff[4]:.2f}')
    plt.plot(bin_centres, gauss(bin_centres, *coeff[6:]),
             c='g', lw=3, ls=':',
             label=f'$\\pi$: mean = {coeff[7]:.2f}')
    # filled in bit for just the electrons
    plt.hist(delta_t[mask_e],
             179, range=(-10, 8),
             histtype='stepfilled', lw=3, alpha=0.2, color='r')
    # add legend, save
    plt.legend()
    plt.savefig("beam_tof_hist.png")
    plt.cla()
    plt.clf()

    # now calculate the offset due to electronics etc
    print("calculating offset")
    # offset should be the same for electrons, muons, and pions, use electrons
    delta_t_electrons = delta_t[mask_e]
    # offset = time taken for electronics to do their thing
    # we assume our particles are really travelling around speed of light, so
    # to get the offset we take our measured time minus time light would take
    offset = delta_t_electrons - length/c_m_per_ns
    offset_mean = np.mean(offset)
    plt.hist(offset, 100)
    plt.axvline(offset_mean, c='k', lw=3)
    plt.xlabel(f"electron offset, ns -- mean = {offset_mean}")
    plt.ylabel("counts")
    plt.savefig("beam_e_offset_hist.png")
    plt.cla()
    plt.clf()

    # for a given momentum we can calculate what our ToF should be
    print("calculating expected ToF")
    momentum = np.arange(50, 300, 1)  # MeV/c

    def get_tof(mass):
        """Get time of flight for a given mass at a bunch of different momenta.
        Mass is in MeV.
        """
        e_total = np.sqrt(momentum**2 + mass**2)
        beta = momentum / e_total
        tof = length / (beta * c_m_per_ns)
        return tof

    electron_mass = 0.511  # MeV
    muon_mass = 105.660  # MeV
    pion_mass = 139.570  # MeV
    calc_e_tof = get_tof(electron_mass)
    calc_mu_tof = get_tof(muon_mass)
    calc_pi_tof = get_tof(pion_mass)

    # our real ToFs (don't need electrons, it should be constant):
    real_mu_tof = m_mean - offset_mean
    real_pi_tof = p_mean - offset_mean

    # best match momentum = where real tof closest to calculated ToF
    p_mu = momentum[np.argmin(abs(calc_mu_tof - real_mu_tof))]
    p_pi = momentum[np.argmin(abs(calc_pi_tof - real_pi_tof))]

    # value to return
    avg_p = np.average([p_mu, p_pi])
    print(f"Calculated beam momentum: {avg_p} MeV/c")

    plt.plot(momentum, calc_e_tof, 'r', label="e")
    plt.plot(momentum, calc_mu_tof, 'orange', label="mu")
    plt.plot(momentum, calc_pi_tof, 'darkgreen', label="pi")
    plt.plot([p_mu, p_pi], [real_mu_tof, real_pi_tof], 'ko')
    plt.plot([avg_p, avg_p], [min(calc_e_tof), max(calc_pi_tof)], 'k')
    plt.title(f"Calculated beam momentum: {avg_p} MeV/c")
    plt.xlabel("Momentum, MeV/c")
    plt.ylabel("Time of flight, ns")
    plt.legend()
    plt.savefig("beam_momentum_vs_tof.png")
    plt.cla()
    plt.clf()

    return avg_p


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "-f", dest="file", default=DEFAULT_ROOT_FILE_NAME,
        help="path to root file with beam data")
    args = parser.parse_args()
    calculate_momentum(args.file)
