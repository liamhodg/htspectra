from scipy.stats import rv_continuous
from scipy.special import hyp1f1, loggamma
from scipy.optimize import fmin
import numpy as np
import mpmath as mpm
from scipy.stats._distn_infrastructure import _ShapeInfo

def _log_neg_hypu(a, b, z):
    hypu_fn = lambda a,b,z: 2*mpm.log(mpm.fabs(
        mpm.hyperu(a, -b, -z)))
    return np.vectorize(hypu_fn, otypes=(float,))(a,b,z)

class marchenko_pastur_gen(rv_continuous):
    """
    Marchenko-Pastur distribution implementation.

    Parameters:
        gam (float): Ratio of dimensions (p / n), where gam > 1.
        scale (float): Scaling factor (default is 1).
    """

    def __init__(self, gam, scale=1):
        c = gam
        if not (0 < c <= 1):
            raise ValueError("Parameter 'gam' must be >= 1.")
        self.c = c
        self.scale = scale
        super().__init__(a=scale * (1 - np.sqrt(c))**2, b=scale * (1 + np.sqrt(c))**2)

    def _pdf(self, x):
        """Probability density function of the Marchenko-Pastur distribution."""
        c, scale = self.c, self.scale
        support_min = scale * (1 - np.sqrt(c))**2
        support_max = scale * (1 + np.sqrt(c))**2

        if np.any(x < support_min) or np.any(x > support_max):
            return 0.0

        sqrt_term = np.sqrt((scale * (1 + np.sqrt(c))**2 - x) * (x - scale * (1 - np.sqrt(c))**2))
        return (1 / (2 * np.pi * c * x * scale)) * sqrt_term

    def _cdf(self, x):
        """Cumulative density function (numerical integration)."""
        from scipy.integrate import quad

        def integrand(t):
            return self._pdf(t)

        result = np.zeros_like(x, dtype=float)
        for i, val in enumerate(x):
            if val > self.a:
                result[i], _ = quad(integrand, self.a, val)
        return result

class _htmp(rv_continuous):
    r"""
    A continuous random variable representing the two-parameter 
    high-temperature Marchenko-Pastur (HTMP) distribution.

    Parameters:
        gam (float): Ratio of dimensions (p / n), where gam > 1.
        kap (float): Kappa shape parameter, where kap > 0.

    Parameters
    ----------
    kappa_max : float, optional
        Maximum allowable value for the `kap` (kappa) parameter to ensure 
        numerical stability. Default is 200.

    Methods
    -------
    logpdf(x, gam, kap)
        Compute the logarithm of the probability density function (PDF).
    pdf(x, gam, kap)
        Compute the probability density function (PDF).
    cdf(x, gam, kap)
        Compute the cumulative distribution function (CDF).
    fit(data)
        Fit the distribution to data.
    stieltjes(x, gam, kap)
        Compute the Stieltjes transform for the distribution.
    """

    def __init__(self, kappa_max=200, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.a = 0
        self.b = np.inf
        self.kappa_max = kappa_max

    def _get_ab(self, gam, kap):
        a = kap/2 * (1 / gam - 1) - 1
        b = kap/2
        return a,b

    def _logpdf(self, x, gam, kap):
        a,b = self._get_ab(gam,kap)
        y = b/gam * x
        const = -loggamma(b+1) - loggamma(a+b+1) + np.log(b/gam)
        return const + a*np.log(y) - y - _log_neg_hypu(b, a, y)
    
    def _logpxf(self, x, gam, kap):
        return self._logpdf(x, gam, kap)
    
    def pdf(self, x, gam, kap):
        return np.exp(self._logpdf(x, gam, kap))

    def _argcheck(self, *args):
        gam = args[0]
        kap = args[1]
        if gam > 1 and kap > 0 and kap < self.kappa_max:
            return True
        return False
    
    def _fitstart(self, data, args=None):
        """Starting point for fit (shape arguments + loc + scale)."""
        if args is None:
            args = (2, 2)
        loc, scale = (0, 1)
        return args + (loc, scale)
    
    def _cdf_mpm(self, x, gam, kap):
        a,b = self._get_ab(gam,kap)
        const = -mpm.loggamma(b+1) - mpm.loggamma(a+b+1) + np.log(b/gam)
        integ = lambda t: (b/gam*t)**a * mpm.exp(-b/gam*t) / \
                    mpm.fabs(mpm.hyperu(b,-a,-b/gam*t))**2
        return mpm.quad(integ, [0, x]) * mpm.exp(const)
    
    def _stieltjes(self, x, gam, kap):
        a,b = self._get_ab(gam,kap)
        numer = mpm.hyperu(b+1,1-a,-x)
        denom = mpm.hyperu(b,-a,-x)
        return numer / denom

    def _cdf(self, x, gam, kap):
        return np.vectorize(self._cdf_mpm, otypes=(float,))(x,gam,kap)
    
    def stieltjes(self, x, gam, kap):
        return np.vectorize(self._stieltjes, otypes=(complex,))(x,gam,kap)
    
    def _shape_info(self):
        return [_ShapeInfo("gam", False, (1,np.inf), (False,False)),
                _ShapeInfo("kap", False, (0,self.kappa_max), (False,False))]


    
htmp = _htmp()