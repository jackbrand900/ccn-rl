# ccn-rl
This project explores how to integrate logical requirements into reinforcement learning (RL) pipelines using the CCN+ framework (https://www.sciencedirect.com/science/article/pii/S0888613X24000112). The goal is to train RL agents that not only maximize reward but also guarantee adherence to formal constraints, improving safety and interpretability in decision-making.

The framework combines:

- Deep RL agents (DQN, PPO, A2C) implemented in PyTorch.

- CCN+ Requirements Layer, a differentiable neuro-symbolic layer enforcing propositional logic constraints, implemented in PiShield.

Shielding is achieved via runtime layers that correct outputs to satisfy rules, during both training and inference.

This setup supports experiments across environments (e.g., Gridworld, CartPole, CarRacing, Atari), evaluating both task performance and constraint satisfaction.
