import gym
from gym import error, spaces, utils
from gym.utils import seeding

import numpy as np
from scipy import optimize
from scipy.integrate import quad
from scipy.stats import norm


class DerivativePricingEnv(gym.Env):
  metadata = {'render.modes': ['human']}

  def __init__(self, params, productParams):

    super(DerivativePricingEnv, self).__init__()

    self.hestonParams, self.assetCorrelations, self.q, self.r = params # hestonParams= V, Vbar, theta, alpha, rho
    self.K, self.period, self.nPeriods, self.cost = productParams

    self.nAssets = self.hestonParams.shape[0]
    self.spotsInitial = np.ones(self.nAssets)
    self.VInitial = self.hestonParams[:, 0]
    self.Vbar = self.hestonParams[:, 1]
    self.theta = self.hestonParams[:, 2]
    self.alpha = self.hestonParams[:, 3]
    self.rho = self.hestonParams[:, 4]

    self.nT = int(1+self.period*365)
    self.dt = self.period/self.nT
    self.sqrtDt = self.dt**0.5

    self.nKs=3
    self.action_space = spaces.Box(low=0, high=1, shape=(self.nAssets,1), dtype=np.float16)
    self.observation_space = spaces.Box(low=-1, high=np.inf,
      shape=(self.nAssets, self.nAssets + 4 + 2 * (1+2*self.nKs)), dtype=np.float16) # correl, basketReturn, time, spot, fwd, strikes, vols


  def step(self, action):
    newWeights = action / np.sum(action)
    self.basketReturn = max(0, self.basketReturn * ( 1 - self.cost*np.sum(np.abs(newWeights-self.weights))))
    self.weights = newWeights

    C = np.log(self.spots)
    for _ in range(self.nT):
        W1=np.random.multivariate_normal(np.zeros(self.nAssets), self.assetCorrelations)
        W2=self.rho*W1+np.sqrt(1-np.power(self.rho,2))*np.random.randn(self.nAssets)
        C=C+np.sqrt(self.V)*W1*self.sqrtDt+(self.r-self.q-0.5*self.V)*self.dt
        self.V=np.maximum(0.0001**2, self.V+self.theta*(self.Vbar-self.V)*self.dt+self.alpha*np.sqrt(self.V)*W2*self.sqrtDt)
    newSpots = np.exp(C)

    self.t += 1
    #self.obsCorrel = self.assetCorrelations.copy()
    self.basketReturn *= np.sum(self.weights * (newSpots / self.spots))
    self.spots = newSpots
    obs = self._next_observation()
    done= self.nPeriods==self.t
    reward = max(self.basketReturn-self.K, 0) if done else 0

    return obs, reward, done, {}


  def reset(self):
    self.spots = self.spotsInitial.copy()
    self.V = self.VInitial.copy()
    self.obsCorrel = self.assetCorrelations.copy()
    self.basketReturn = 1
    self.t = 0
    self.weights = np.ones(self.nAssets)/self.nAssets

    return self._next_observation()


  def render(self, mode='human'):
    obs = self._next_observation()
    print(obs)

  def close(self):
    pass


  def _next_observation(self):
    obs=np.zeros((self.nAssets,self.nAssets + 4 + 2 * (1+2*self.nKs)))
    for i in range(self.nAssets):
      obs[i,:self.nAssets]=self.obsCorrel[i,:]
      obs[i,self.nAssets]=self.basketReturn
      obs[i,self.nAssets+1]=self.t
      obs[i,self.nAssets+2]=self.spots[i]
      fwd = self.spots[i]*np.exp((self.r - self.q[i]) * self.period)
      obs[i,self.nAssets+3]=fwd
      Ks = [fwd*np.exp(j*self.V[i]*self.period**0.5) for j in range(-self.nKs,self.nKs+1)]
      for j in range(len(Ks)):
        obs[i, self.nAssets + 4 + j] = Ks[j]
        price = heston_put(self.spots[i], Ks[j], self.period, self.V[i],self.Vbar[i],self.theta[i],self.alpha[i],self.rho[i],self.r,self.q[i]) if Ks[j] < fwd \
          else heston_call(self.spots[i], Ks[j], self.period, self.V[i],self.Vbar[i],self.theta[i],self.alpha[i],self.rho[i],self.r,self.q[i])
        iv=implied_vol(price * np.exp(self.r * self.period), fwd, Ks[j], self.period, Ks[j]>=fwd)
        obs[i, self.nAssets + 4 + j + 2*self.nKs+1] = iv
    return obs



# finance functions

def bsprice(S, K, T, sigma, isCall=True):
    d1 = (np.log(S / K) + (0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    if isCall:
      return S * norm.cdf(d1, 0, 1) - K * norm.cdf(d2, 0, 1)
    else:
      return K * norm.cdf(-d2, 0, 1) - S * norm.cdf(-d1, 0, 1)


def implied_vol(target_value, S, K, T, isCall=True):
  return optimize.brentq(lambda sig: bsprice(S, K, T, sig, isCall) - target_value, 0.01, 0.75)


def heston_phi(z, tau, v, vbar, lamb, eta, rho):
  a0 = lamb - rho * eta * z * 1j
  gamma = np.power(eta ** 2 * (z ** 2 + z * 1j) + a0 * a0, 0.5)
  G = (a0 - gamma) / (a0 + gamma)
  a1 = v / eta / eta * ((1 - np.exp(-gamma * tau)) / (1 - G * np.exp(-gamma * tau))) * (a0 - gamma)
  a2 = lamb * vbar / eta / eta * (tau * (a0 - gamma) - 2 * np.log((1 - G * np.exp(-gamma * tau)) / (1 - G)))  # c*vbar
  return np.exp(a1 + a2)


def heston_phi_call(k, tau, v, vbar, lambd, eta, rho, AL):
  integrand = lambda z: (np.exp(-z * k * 1j) * heston_phi(z - (AL + 1) * 1j, tau, v, vbar, lambd, eta, rho) / (
            AL ** 2 + AL - z ** 2 + 1j * (2 * AL + 1) * z)).real
  return quad(integrand, 0, 500, limit=250)[0]


def heston_call(S, K, tau, v, vbar, lambd, eta, rho, r=0, divYield=0):
  dampFac = 0.5
  k = np.log(K / S) - r * tau + divYield * tau
  integral = heston_phi_call(k, tau, v, vbar, lambd, eta, rho, dampFac)
  return np.exp(-divYield * tau) * S * np.exp(-dampFac * k) * integral / np.pi


def heston_phi_put(k, tau, v, vbar, lambd, eta, rho, AL):
  integrand = lambda z: (np.exp(-z * k * 1j) * heston_phi(z - (-AL + 1) * 1j, tau, v, vbar, lambd, eta, rho) / (
            AL ** 2 - AL - z ** 2 + 1j * (-2 * AL + 1) * z)).real
  return quad(integrand, 0, 500, limit=250)[0]


def heston_put(S, K, tau, v, vbar, lambd, eta, rho, r=0, divYield=0):
  dampFac = 3
  k = np.log(K / S) - r * tau + divYield * tau
  integral = heston_phi_put(k, tau, v, vbar, lambd, eta, rho, dampFac)
  return np.exp(-divYield * tau) * S * np.exp(dampFac * k) * integral / np.pi
