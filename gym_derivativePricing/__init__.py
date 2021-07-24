from gym.envs.registration import register

register(
    id='derivativePricing-v0',
    entry_point='gym_derivativePricing.envs:DerivativePricingEnv',
)