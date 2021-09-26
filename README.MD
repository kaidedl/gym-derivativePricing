The derivativePricing environment is a single-agent
domain featuring continuous state and action spaces in a stochastic model. The task is to rebalance periodically a portfolio of n stocks in order to maximize the value of a call option on this arithmetic basket. There is a (hedge) cost associated with each rebalancing.

The feature space includes spots, forwards, correlations and implied vol surfaces. An observation is a matrix where each row corresponds to an asset and contains the correlations to the other assets, the basket return, the number of remaining time steps, the spot, the forward, three strikes around the forward and the corresponding vols. The forwards and vols are computed for one time period.

The model is the Heston model.

The action space is given by weights 0<x_i<1 for asset i such that x_i / (x_1+..+x_n) is invested in asset i. Hence an action is given by a real valued vector with size n corresponding to the number of assets.

# Installation

```bash
cd gym-derivativePricing
pip install -e .
```

# Example: Uniform random policy

```bash
def policy_random(obs):
  alpha=np.ones(obs.shape[0])
  return np.random.dirichlet(alpha)
```

# Example: Constant policy

```bash
def policy_const(obs):
  return np.ones(obs.shape[0])/obs.shape[0]
```

# Example: Approximative policy ignoring cost

```bash
import gym

import numpy as np
from tqdm import tqdm
from scipy import interpolate
from scipy.stats import norm


def policy_approx(obs):
    n=obs.shape[0]
    basketValue = obs[0,n]
    noRemainingPeriods = obs[0,n+1]

    vols=[]
    fwds=[]
    for i in range(n):
        Ksi = obs[i,n+4:n+7]
        volsi = obs[i,n+7:n+10]
        smile=interpolate.interp1d(Ksi, volsi, fill_value="extrapolate")
        voli=smile((1+(K/basketValue-1)/noRemainingPeriods)*obs[i,n+2])
        vols.append(float(voli))
        fwds.append(obs[i,n+3]/obs[i,n+2])

    def basketOptionValue(w, fwds, C, K, period):
        w=w/np.sum(w)
        fwd = np.sum(w*fwds)
        w=w*vols
        vol = np.sqrt(np.matmul(w,np.matmul(C,w)))
        d1 = (np.log(fwd / K) + (0.5 * vol ** 2) * period) / (vol * np.sqrt(period))
        d2 = d1 - vol * np.sqrt(period)
        return -(fwd * norm.cdf(d1, 0, 1) - K * norm.cdf(d2, 0, 1))

    C=obs[:n,:n]
    p=np.ones(n)/n
    dp=0.1*np.ones(n)
    best_err=basketOptionValue(p,fwds,C,K,period)
    while(np.sum(dp))>0.01:
        for i in range(n):
            dp[i]=min(dp[i],min(1-p[i],p[i])/2)
            p[i]+=dp[i]
            err=basketOptionValue(p,fwds,C,K,period)
            if err<best_err:
                best_err=err
            else:
                p[i] -= 2*dp[i]
                err=basketOptionValue(p,fwds,C,K,period)
                if err<best_err:
                    best_err=err
                else:
                    p[i] += dp[i]
                    dp[i]*=0.9

    return p/np.sum(p)


if __name__ == '__main__':
    hestonParams=np.array([[0.04,0.04,3,0.3,-0.5],
            [0.04,0.04,3,0.3,-0.6],
            [0.04, 0.06, 3, 0.3, -0.7]])
    corr = np.array([[1,0.8,0.3],
            [0.8,1,0.7],
            [0.3, 0.7, 1]])
    q=np.array([0.03,0.04, 0.02])
    r=0.02
    K=1
    period=0.25
    nPeriods=8

    env=gym.make('gym_derivativePricing:derivativePricing-v0',
                 params=[hestonParams, corr, q, r],
                 productParams=[K, period, nPeriods, 0.01])

    rewards=[]
    numSims=1
    np.random.seed(1000)
    for _ in tqdm(range(numSims)):
        obs = env.reset()
        done=False
        while not done:
            action = policy_approx(obs)
            obs, reward, done, info = env.step(action)
        rewards.append(reward)
    print(reward)
```
