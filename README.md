# DRL.jl

## Deep Reinforcement Learning for Julia

It's actually not so gradiose. 

Julia implementations of deep reinforcement learning algorithms. Uses the [POMDPs.jl](https://github.com/JuliaPOMDP/POMDPs.jl) framework for representing (Partially Observable) Markov Decision Proccesses (which itself is a framework for describing sequential decision making problems). 

Working-ish:
* Deep Q Learning (DQN)

Currently working on:
* Model-Free Episodic Control (QEC)
* Deterministic Policy Gradient (DDPG)

To work on:
* Stochastic Policy Gradient (SDPG)
* Trust Region Policy Optimization (TRPO)
* Trust Region Generalized Advantage Estimate (TRGAE)
* VIME
* Misc NN models for MXNet (GAN, VAE, RNN)

NOTE: the signature for solve doesn't exactly match `POMDPs.solve`. It is `solve(::Solver, ::MDP, ::Policy, ::rng)
NOTE: try to define your vectors as Float32 if possible (which is what mxnet uses)

Documentation someday
