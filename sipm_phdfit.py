# -*- coding: utf-8 -*-
"""
Created on Wed Jan 04 15:53:04 2017
11/01/2017
    add_oc - changed xtalk dependence from 2*sqrt(i) i
12/01/2017
    removed round() in add_oc
    improved hparm() to allow bins, range[0] and binwidth to define
    thresh set to 0 - threshing only necessary for prior processed area data

@author: Jon
"""

import numpy as np
import matplotlib.pyplot as plt
from utils.draw import draw_gaussian
from scipy.special import factorial
from scipy.signal import find_peaks
from lmfit import Model, Parameters

def draw_gaussian(x, area, centre, sigma):
    """
    Generate a Gaussian distribution with area = 1.

    Parameters
    ----------
    x : array
        An array of the x coordinates
    area : scalar int
        The area under the Gaussian = the number of events constituting
    centre : scalar
        The centre of the Gaussian
    sigma : scalar
        The standard deviation

    Returns
    -------
    data : ndarray 
        The Gaussian distribution

    """
    step = (x[-1] - x[0])/(len(x) - 1)
    s2pi = np.sqrt(2.0*np.pi)
    return (area * step) / (sigma * s2pi) * np.exp(-1.0*(x-centre)**2 /(2*sigma**2))

def poisson(mu, arrsz):
    """
    Array of Poisson probabilities for a given mean number per event.

    Parameters
    ----------
    mu : float scalar
        The mean number per event
    arrsz : int scalar
        The array size to return probabilities

    Returns
    -------
    pdist : ndarray 
        Discrete Poisson distribution of size 'arrsz'.

    """
    pts = range(arrsz)
    mu = np.ones_like(pts)*mu
    pdist = np.exp(-mu) * np.power(mu,pts) / factorial(pts)
    return pdist

def genpoisson(mu, lmbda, n):
    """
    Array of Generalized Poisson probabilities for a given mean number per event and per xtalk event.

    Parameters
    ----------
    mu : float scalar
        The mean number per event
    lmbda : float scalar
        The mean number per xtalk event
    n : int scalar
        The array size to return probabilities

    Returns
    -------
    gpdist : ndarray 
        Generalized Poisson distribution of size 'n'.

    """
    k = np.arange(n)
    mu = np.ones_like(k, dtype=np.float)*mu
    gpdist = mu * np.power(mu+k*lmbda,k-1) * np.exp(-mu-k*lmbda) / factorial(k) 
    return gpdist

# phd fitting function
def sipm_fitfunc(x, xoff, mu, nev, xtalk, v_pe, v_n, v_gain, thresh=0, nz_pe=0):
    arrsz = max(int(mu*10), 30)
    p = poisson(mu, arrsz)
    gp = genpoisson(mu, xtalk, arrsz)
    if nz_pe == 1: # exclude the 0 pe peak
        gp[0] = 0 # in case there's no pedestal data i.e. triggered only on events
    #sipm_dist = gp*nev*p[1]/gp[1] # normalise gp[1] - 1 pe peak has no xtalk --> OCT increases average number of pe
    sipm_dist = gp*nev# don't normalise
    output = np.zeros_like(x)
    for i in range(len(sipm_dist)):
        gausswid = np.sqrt(v_n**2 + i*v_gain**2)
        output += draw_gaussian(x, sipm_dist[i], i*v_pe+xoff, gausswid)
    output[0:thresh] = 0
    return output

def sipm_phdfit(x, y, npk, nz_pe=0):
    """
    Function to fit a SiPM phd array with a Generalized Poisson 

    Parameters
    ----------
    x : float array
        Typically the pulse heigth in mV
    y : int array
        The number of events per bin
    npk : int scalar
        The number of distinguishable peaks in the PHD - for initial parameter guess
    nz_pe : If 0 - include pedestal in PHD
            If 1 - exclude pedestal

    Returns
    -------
    result : lmfit result object 
        Including initial guess, best fit, and all fit parameters

    """
    ymax = y.max()
    #find peaks in the PHD to npk
    for i in range(50):
        peaks, p_prop = find_peaks(y, prominence=ymax*(1-i/50), height=ymax/10)
        if len(peaks)>=npk: break
    #now estimate the initial fit parameters
    mu = np.sum(p_prop['peak_heights']*np.arange(nz_pe,npk+nz_pe))/np.sum(p_prop['peak_heights'])
    nev = np.sum(y)
    xtalk = 0.5 #based on CHEC-S devices
    v_pe = np.mean(np.diff(x[peaks]))
    xoff = x[peaks[0]] - v_pe*nz_pe
    v_n = v_pe*0.2
    v_gain = v_pe*0.1
    thresh = 0

    smod  = Model(sipm_fitfunc)
    pars = Parameters()
    # parameter constraints
    #           (Name,     Value,  Vary,   Min,  Max,  Expr, Brute Step)
    pars.add_many(('xoff',  xoff,  True,-50.0,  50.0,  None, None),
               ('mu',         mu,  True, 0.01,  50.0,  None, None),
               ('nev',       nev,  True,    1,   1e8,  None, None),
               ('xtalk',   xtalk,  True,  0.0,  0.75,  None, None),
               ('v_pe',     v_pe,  True,  0.0,  50.0,  None, None),
               ('v_n',       v_n,  True,  0.0,  50.0,  None, None),
               ('v_gain', v_gain,  True,  0.0,  50.0,  None, None),
               ('thresh', thresh,  False,   0,   500,  None, None),
               ('nz_pe',   nz_pe,  False,   0,     1,  None, None))
    #solve 
    result =  smod.fit(y, params=pars, x=x, method='leastsq')
    return result

def main():
    #a simple test procedure
    filename = './set2_67.4_C1_peaks.npz'
    npzfile = np.load(filename)
    x, y = npzfile['arr_0'], npzfile['arr_1']
    #input number of distinguishable peaks
    npk = 4
    result = sipm_phdfit(x, y, npk, nz_pe=1)
    # plot measured data
    plt.plot(x, y, 'b', linewidth=1, label='raw data')
    plt.plot(x, result.init_fit, 'k-', linewidth=1, label='guess')
    plt.plot(x, result.best_fit, 'r-', linewidth=1, label='best fit')
    plt.legend(loc='best')
    plt.show()
    #save fit parameters file
    datafile = open(initdir+"\phd_fit_params.txt", 'a')
    datafile.write('{0}, '.format(filename))
    datafile.write('{0}, '.format(result.params['xoff'].value))
    datafile.write('{0}, '.format(result.params['mu'].value))
    datafile.write('{0}, '.format(result.params['nev'].value))
    datafile.write('{0}, '.format(result.params['xtalk'].value))
    datafile.write('{0}, '.format(result.params['v_pe'].value))
    datafile.write('{0}, '.format(result.params['v_n'].value))
    datafile.write('{0}, '.format(result.params['v_gain'].value))
    datafile.write('{0}, '.format(result.params['thresh'].value))
    datafile.write('{0}, '.format(result.params['nz_pe'].value))
    datafile.write('{0}, '.format(result.redchi))
    datafile.write('{0}, '.format(np.sum(y)))
    datafile.write('{0}, '.format(np.sum(result.best_fit)))
    datafile.close()
    #print result
    print(result.fit_report())  

if __name__ == '__main__':
    main()
