[//]: # (Image References)

[image1]: https://user-images.githubusercontent.com/10624937/42135623-e770e354-7d12-11e8-998d-29fc74429ca2.gif "Trained Agent"
[image2]: https://user-images.githubusercontent.com/10624937/42135622-e55fb586-7d12-11e8-8a54-3c31da15a90a.gif "Soccer"


# Project 3: Collaboration and Competition

### Introduction

For this project, you will work with the [Tennis](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Learning-Environment-Examples.md#tennis) environment.

![Trained Agent][image1]

In this environment, two agents control rackets to bounce a ball over a net. If an agent hits the ball over the net, it receives a reward of +0.1.  If an agent lets a ball hit the ground or hits the ball out of bounds, it receives a reward of -0.01.  Thus, the goal of each agent is to keep the ball in play.

The observation space consists of 8 variables corresponding to the position and velocity of the ball and racket. Each agent receives its own, local observation.  Two continuous actions are available, corresponding to movement toward (or away from) the net, and jumping. 

The task is episodic, and in order to solve the environment, your agents must get an average score of +0.5 (over 100 consecutive episodes, after taking the maximum over both agents). Specifically,

- After each episode, we add up the rewards that each agent received (without discounting), to get a score for each agent. This yields 2 (potentially different) scores. We then take the maximum of these 2 scores.
- This yields a single **score** for each episode.

The environment is considered solved, when the average (over 100 episodes) of those **scores** is at least +0.5.

### Getting Started

The provided Dockerfile should be used to build the Python execution environment. Note that this environment is based on Nvidia CUDA image so should execute via GPU on any NVidia GPU-compatible Docker system. Also note that this serves as a *remote* execution environment - i.e. the resultant Docker image will not contain the executable Python scripts, but these Python scripts can be run on the execution environment. Please refer to your IDE for more details on how to enable this in your setup. 

Additionally, note that the included Reacher environments are strictly the headless Linux ones intended for AWS in the documentation, since these are the ones which are compatible with Docker across all of the systems where this may be executed.

In order to train the agent, please run the train.py file using the environment described above. Note that the environment comes provisioned with Weights and Biases so if the --prod-mode and --wandb-api-key command line arguments are passed in, then the training session will be uploaded to wandb. This is the best way to produce the plot of rewards per episode for the training session
