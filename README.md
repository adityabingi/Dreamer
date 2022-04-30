# Dreamer

Dreamer is a visual Model-Based Reinforcement algorithm, that learns a world model that captures latent dynamics from high-level pixels and trains a control agent entirely in imagined rollouts from the learned world model.

This work is my attempt at reproducing Dreamerv1 & v2 papers in pytorch specifically for continuous control tasks in deepmind control suite.

#### Noteworthy differences from original and prior works:

 1. This work compares Dreamer and Dreamerv2 agents for continuous control tasks only. Only KL-Balancing is used for dreamerv2 and policy type remains the same as dreamerv1 i.e., tanh distribution policy.
 2. This work doesn't train dreamerv1 and v2 for 2M frames as did in the papers, instead both the agents are trained till 100K frames.
 3. All experiments are carried out on free single GPUs(Tesla T4) on google colab. Training time on Tesla T4 for 100K frames ~ 3 Hrs
 4. Due to limited computational resources (colab strict timeouts) results produced here are for five control tasks and are run for single seed only. 
 5. Hence plot_resultsare produced by running agents for 10 eval episodes for single seed. A fair evaluation would require running experiments for multiple seeds, this repo serves as a working implementation for both agents.

Evaluated agents are shown below Left to Right (cartpole-balance, walker-stand, cartpole-swingup, walker-walk, cheetah-run) after training till 100K frames  

|Dreamerv1 | Dreamerv2 |
| ---| ---|
|![dreamer](results/dreamer.gif)|![dreamerv2](results/dreamerv2.gif)|

## Algorithm

For further information regarding methodology and experiments refer these papers
1. [Dreamerv1 - DREAM TO CONTROL: LEARNING BEHAVIORS BY LATENT IMAGINATION](https://arxiv.org/pdf/1912.01603.pdf)
2. [Dreamerv2 - MASTERING ATARI WITH DISCRETE WORLD MODELS](https://arxiv.org/pdf/2010.02193.pdf)

## Code Structure
Code structure is similar to original work by Danijar Hafner in Tensorflow

dreamer.py  - main function for training and evaluating dreamer agent

utils.py    - Logger, 

models.py   - All the networks are implemented here

replay_buffer.py - Experience buffer for training World Model

env_wrapper.py  - Gym wrapper for Dm_control suite

#### For training
`python dreamer.py --env 'walker-walk' --algo 'Dreamerv1' --exp 'default_hp' --train`

#### For Evaluation
`python dreamer.py --env 'walker-walk' --algo 'Dreamerv1' --exp 'eval' --evaluate --restore --checkpoint_path '<your_ckpt_path>'`
## Plot Results

## Acknowledgements
This code is heavily inpsired by following open-source works

dreamer by Danijar Hafner(lead author of both papers) : https://github.com/danijar/dreamer/blob/master/dreamer.py

dreamer-pytorch by yusukeurakami : https://github.com/yusukeurakami/dreamer-pytorch

Dreamerv2 by Rajghugare : https://github.com/RajGhugare19/dreamerv2
