from __future__ import (absolute_import, division,
                        print_function)
import numpy as np
import matplotlib.pyplot as plt
from glob import glob
import json
import os
import hashlib

try:
    import cPickle as pickle
except:
    import pickle

from enterprise.pulsar import Pulsar
from PTMCMCSampler.PTMCMCSampler import PTSampler as ptmcmc


DATADIR = 'nano11y_data/'
CACHEDIR = '.data_cache/'
os.system('mkdir -p {}'.format(CACHEDIR))

def get_pulsars(psrlist=None, ephem='DE436', use_cache=True):
    """
    Reads in list of pulsar names and ephemeris version
    and returns list of instantiated enterprise Pulsar objects.

    By default the input list is None and we use the 34 pulsars used in
    the stochastic background analysis. The default ephemeris is that used
    in the 11-year dataset (DE436)
    """
    if psrlist is None:
        psrlist = np.loadtxt('psrlist.txt', dtype=np.unicode_)

    parfiles = sorted(glob(DATADIR+'/partim/*.par'))
    timfiles = sorted(glob(DATADIR+'/partim/*.tim'))

    # create pulsar hash
    psr_str = ''.join(sorted(psrlist)) + ephem
    psr_hash = hashlib.sha1(psr_str.encode()).hexdigest()

    # check for cached file
    cached_file = CACHEDIR + psr_hash
    if os.path.exists(cached_file) and use_cache:
        print('Reading pulsars from cached file.\n')
        with open(cached_file, 'rb') as fin:
            psrs = pickle.load(fin)
    else:
        psrs = []
        for par, tim in zip(parfiles, timfiles):
            pname = par.split('/')[-1].split('_')[0]
            if pname in psrlist:
                psrs.append(Pulsar(par, tim, ephem=ephem))

        print('Writing pulsars to cache.\n')
        with open(cached_file, 'wb') as fout:
            pickle.dump(psrs, fout)

    return psrs

def get_noise_dict(psrlist=None):
    """
    Reads in list of pulsar names and returns dictionary
    of {parameter_name: value} for all noise parameters.

    By default the input list is None and we use the 34 pulsars used in
    the stochastic background analysis.
    """

    if psrlist is None:
        psrlist = np.loadtxt('psrlist.txt', dtype=np.unicode_)

    params = {}
    for p in psrlist:
        with open(DATADIR+'/noisefiles/{}_noise.json'.format(p), 'r') as fin:
            params.update(json.load(fin))

    return params

class JumpProposal(object):

    def __init__(self, pta):
        """Set up some custom jump proposals"""
        self.params = pta.params
        self.pnames = pta.param_names
        self.npar = len(pta.params)
        self.ndim = sum(p.size or 1 for p in pta.params)

        # parameter map
        self.pmap = {}
        ct = 0
        for p in pta.params:
            size = p.size or 1
            self.pmap[p] = slice(ct, ct+size)
            ct += size

        # parameter indices map
        self.pimap = {}
        for ct, p in enumerate(pta.param_names):
            self.pimap[p] = ct

        self.snames = {}
        for sc in pta._signalcollections:
            for signal in sc._signals:
                self.snames[signal.signal_name] = signal.params

    def draw_from_prior(self, x, iter, beta):
        """Prior draw.

        The function signature is specific to PTMCMCSampler.
        """

        q = x.copy()
        lqxy = 0

        # randomly choose parameter
        idx = np.random.randint(0, self.npar)

        # if vector parameter jump in random component
        param = self.params[idx]
        if param.size:
            idx2 = np.random.randint(0, param.size)
            q[self.pmap[param]][idx2] = param.sample()[idx2]

        # scalar parameter
        else:
            q[idx] = param.sample()

        # forward-backward jump probability
        lqxy = param.get_logpdf(x[self.pmap[param]]) - param.get_logpdf(q[self.pmap[param]])

        return q, float(lqxy)

    def draw_from_gwb_prior(self, x, iter, beta):

        q = x.copy()
        lqxy = 0

        signal_name = 'red noise'

        # draw parameter from signal model
        param = np.random.choice(self.snames[signal_name])
        if param.size:
            idx2 = np.random.randint(0, param.size)
            q[self.pmap[param]][idx2] = param.sample()[idx2]

        # scalar parameter
        else:
            q[self.pmap[param]] = param.sample()

        # forward-backward jump probability
        lqxy = param.get_logpdf(x[self.pmap[param]]) - param.get_logpdf(q[self.pmap[param]])

        return q, float(lqxy)

    def draw_from_gwb_log_uniform_distribution(self, x, iter, beta):

        q = x.copy()
        lqxy = 0

        # draw parameter from signal model
        idx = self.pnames.index('log10_A_gw')
        q[idx] = np.random.uniform(-18, -11)

        return q, 0

    def draw_from_dipole_log_uniform_distribution(self, x, iter, beta):

        q = x.copy()
        lqxy = 0

        # draw parameter from signal model
        idx = self.pnames.index('log10_A_dipole')
        q[idx] = np.random.uniform(-18, -11)

        return q, 0

    def draw_from_monopole_log_uniform_distribution(self, x, iter, beta):

        q = x.copy()
        lqxy = 0

        # draw parameter from signal model
        idx = self.pnames.index('log10_A_monopole')
        q[idx] = np.random.uniform(-18, -11)

        return q, 0

    def draw_from_ephem_prior(self, x, iter, beta):

        q = x.copy()
        lqxy = 0

        signal_name = 'phys_ephem'

        # draw parameter from signal model
        param = np.random.choice(self.snames[signal_name])
        if param.size:
            idx2 = np.random.randint(0, param.size)
            q[self.pmap[param]][idx2] = param.sample()[idx2]

        # scalar parameter
        else:
            q[self.pmap[param]] = param.sample()

        # forward-backward jump probability
        lqxy = param.get_logpdf(x[self.pmap[param]]) - param.get_logpdf(q[self.pmap[param]])

        return q, float(lqxy)


def get_global_parameters(pta):
    """Utility function for finding global parameters."""
    pars = []
    for sc in pta._signalcollections:
        pars.extend(sc.param_names)

    gpars = np.unique(list(filter(lambda x: pars.count(x)>1, pars)))
    ipars = np.array([p for p in pars if p not in gpars])

    return gpars, ipars


def get_parameter_groups(pta):
    """Utility function to get parameter groupings for sampling."""
    ndim = len(pta.param_names)
    groups  = [range(0, ndim)]
    params = pta.param_names

    # get global and individual parameters
    gpars, ipars = get_global_parameters(pta)
    if any(gpars):
        groups.extend([[params.index(gp) for gp in gpars]])

    for sc in pta._signalcollections:
        for signal in sc._signals:
            ind = [params.index(p) for p in signal.param_names if p not in gpars]
            if ind:
                groups.extend([ind])

    return groups


def setup_sampler(pta, outdir='chains', resume=False):
    """
    Sets up an instance of PTMCMC sampler.

    We initialize the sampler the likelihood and prior function
    from the PTA object. We set up an initial jump covariance matrix
    with fairly small jumps as this will be adapted as the MCMC runs.

    We will setup an output directory in `outdir` that will contain
    the chain (first n columns are the samples for the n parameters
    and last 4 are log-posterior, log-likelihood, acceptance rate, and
    an indicator variable for parallel tempering but it doesn't matter
    because we aren't using parallel tempering).

    We then add several custom jump proposals to the mix based on
    whether or not certain parameters are in the model. These are
    all either draws from the prior distribution of parameters or
    draws from uniform distributions.
    """

    # dimension of parameter space
    ndim = len(pta.param_names)

    # initial jump covariance matrix
    cov = np.diag(np.ones(ndim) * 0.1**2)

    # parameter groupings
    groups = get_parameter_groups(pta)

    sampler = ptmcmc(ndim, pta.get_lnlikelihood, pta.get_lnprior, cov, groups=groups,
                     outDir=outdir, resume=resume)
    np.savetxt(outdir+'/pars.txt',
               list(map(str, pta.param_names)), fmt='%s')
    np.savetxt(outdir+'/priors.txt',
               list(map(lambda x: str(x.__repr__()), pta.params)), fmt='%s')

    # additional jump proposals
    jp = JumpProposal(pta)

    # always add draw from prior
    sampler.addProposalToCycle(jp.draw_from_prior, 5)

    # Ephemeris prior draw
    if 'd_jupiter_mass' in pta.param_names:
        print('Adding ephemeris model prior draws...\n')
        sampler.addProposalToCycle(jp.draw_from_ephem_prior, 10)

    # GWB uniform distribution draw
    if 'log10_A_gw' in pta.param_names:
        print('Adding GWB uniform distribution draws...\n')
        sampler.addProposalToCycle(jp.draw_from_gwb_log_uniform_distribution, 10)

    # Dipole uniform distribution draw
    if 'log10_A_dipole' in pta.param_names:
        print('Adding dipole uniform distribution draws...\n')
        sampler.addProposalToCycle(jp.draw_from_dipole_log_uniform_distribution, 10)

    # GWB uniform distribution draw
    if 'log10_A_monopole' in pta.param_names:
        print('Adding monopole uniform distribution draws...\n')
        sampler.addProposalToCycle(jp.draw_from_monopole_log_uniform_distribution, 10)

    return sampler

class PostProcessing(object):

    def __init__(self, chain, pars, burn_percentage=0.25):
        burn = int(burn_percentage*chain.shape[0])
        self.chain = chain[burn:]
        self.pars = pars

    def plot_trace(self, plot_kwargs={}):
        ndim = len(self.pars)
        if ndim > 1:
            ncols = 4
            nrows = int(np.ceil(ndim/ncols))
        else:
            ncols, nrows = 1,1

        fig = plt.figure(figsize=(15, 2*nrows))
        for ii in range(ndim):
            plt.subplot(nrows, ncols, ii+1)
            plt.plot(self.chain[:, ii], **plot_kwargs)
            plt.title(self.pars[ii], fontsize=8)
        plt.tight_layout()

    def plot_hist(self, hist_kwargs={'bins':50, 'normed':True}):
        ndim = len(self.pars)
        if ndim > 1:
            ncols = 4
            nrows = int(np.ceil(ndim/ncols))
        else:
            ncols, nrows = 1,1

        fig = plt.figure(figsize=(15, 2*nrows))
        for ii in range(ndim):
            plt.subplot(nrows, ncols, ii+1)
            plt.hist(self.chain[:, ii], **hist_kwargs)
            plt.title(self.pars[ii], fontsize=8)
        plt.tight_layout()


def bayes_fac(samples, ntol=200):
    """
    Computes the Savage Dickey Bayes Factor and uncertainty.

    :param samples: MCMC samples of GWB (or common red noise) amplitude
    :param ntol: Tolerance on number of samples in bin

    :returns: (bayes factor, 1-sigma bayes factor uncertainty)
    """

    logAmin = -18
    logAmax = -14

    prior = 1 / (logAmax - logAmin)
    dA = np.linspace(0.01, 0.1, 100)
    bf = []
    bf_err = []
    mask = [] # selecting bins with more than 200 samples

    for ii,delta in enumerate(dA):
        n = np.sum(samples <= (logAmin + delta))
        N = len(samples)

        post = n / N / delta

        bf.append(prior/post)
        bf_err.append(bf[ii]/np.sqrt(n))

        if n > ntol:
            mask.append(ii)

    return np.mean(np.array(bf)[mask]), np.std(np.array(bf)[mask])
