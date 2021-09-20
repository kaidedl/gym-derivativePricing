The derivativePricing environment is a single-agent
domain featuring continuous state and action spaces. The task is to rebalance periodically a portfolio of n stocks in order to maximize the value of a call option on this arithmetic basket. There is a (hedge) cost associated with each rebalancing.

The feature space includes spots, forwards, correlations and implied vol surfaces. An observation is a matrix where each row corresponds to an asset and contains the correlations, the basket return, the time step, the spot, the forward, some strikes and the corresponding vols. The forwards and vols are computed for one time period.

The action space is given by weights 0<x_i<1 for asset i such that x_i / (x_1+..+x_n) is invested in asset i. Hence an action is given by a real valued vector with size n corresponding to the number of assets.

# Installation

```bash
cd gym-derivativePricing
pip install -e .
```

# Example: Uniform Dirichlet policy

```bash
import gym
import numpy as np


def policy_random(obs):
  alpha=np.ones(obs.shape[0])
  return np.random.dirichlet(alpha)


if __name__ == '__main__':
    hestonParams=np.array([[0.04,0.04,3,0.3,-0.5],
            [0.04,0.09,3,0.3,-0.5]])
    corr = np.array([[1,0.8],
            [0.8,1]])
    q=np.array([0.03,0.04])
    r=0.02
    K=1
    period=0.25
    nPeriods=8

    env=gym.make('gym_derivativePricing:derivativePricing-v0',
                 params=[hestonParams, corr, q, r],
                 productParams=[K, period, nPeriods, 0])

    obs = env.reset()
    done=False
    while not done:
        action = policy_random(obs)
        print(action)
        obs, rewards, done, info = env.step(action)
        env.render()
        print
    print("reward", rewards)
```
