# UdacityMultiAgentDDPGProject2
Multi agent DDPG project Udacity Deep Reinforcement learning nano degree

## Introduction 

This is the third and final project of the Udacity Deep Reinforcement Learning course. In this project Udacity provides a Unity3D application that is used as a training environment with MADDPG (Multi Agent Deep Deterministic Policy Gradient). The goal of this environment is for the agents to play tennis and keep the ball in the air. The environment provides two agents, eacht with their own observations

```python
Number of agents: 2
Number of actions: 2
There are 2 agents. Each observes a state with length: 24
The state for the first agent looks like:  [ 0.          0.          0.          0.          0.          0.
  0.          0.          0.          0.          0.          0.
  0.          0.          0.          0.         -6.65278625 -1.5
 -0.          0.          6.83172083  6.         -0.          0.        ]
```

The agents try to find the action with the most future cumulative reward, and thus trains the deep Neural network to predict the best action, given a random state.

{UPDAAAATE}
[![Training Youtube video](https://img.youtube.com/vi/c-2tIOe-1K4/0.jpg)](https://www.youtube.com/watch?v=c-2tIOe-1K4). 

*Training in progress not wasting any GPU cycles on rendering a full size image*

{UPDAAAATE}
[![Playing Youtube video](https://img.youtube.com/vi/sxyRiDX6dzE/0.jpg)](https://www.youtube.com/watch?v=sxyRiDX6dzE)

*Tennis agents*


## Setup the environment in Windows 10

As with most machine learning projects, its best to start with setting up a virtual environment. This way the packages that need to be imported, don't conflict with Python packages that are already installed. For this project I used the Anaconda environment based on Python 3.6. 

While the example project provides a requirements.txt, I ren into this error while adding the required packages to your project

```python
!pip -q install ./
ERROR: Could not find a version that satisfies the requirement torch==0.4.0 (from unityagents==0.4.0) (from versions: 0.1.2, 0.1.2.post1, 0.1.2.post2)
ERROR: No matching distribution found for torch==0.4.0 (from unityagents==0.4.0)
```

The solution, is to install a downloaded wheel file form the PyTorch website yourself. I downloaded "torch-0.4.1-cp36-cp36m-win_amd64.whl" from the PyTorch site https://pytorch.org/get-started/previous-versions/

```
(UdacityRLProject1) C:\Clients\Udacity\deep-reinforcement-learning\[your project folder]>pip install torch-0.4.1-cp36-cp36m-win_amd64.whl
Processing c:\clients\udacity\deep-reinforcement-learning\[your project folder]\torch-0.4.1-cp36-cp36m-win_amd64.whl
Installing collected packages: torch
Successfully installed torch-0.4.1
```

When all dependencies are resolved, the training can begin.

## Training the agents with code provided by the course

To start, and make sure the environment works, I used the DDPG example from the previous example, and just added two agents with their own actor and critic network. As expected, from what i learned about multi agent training, this didn't perform at all. 

{UPDAAAATE}
![Training with default epsilon decay](https://github.com/fuzzballb/UdacityDDPGProject2/blob/master/images/FailedToLearn.PNG "Training with default epsilon decay")


## Solutions for getting a better score





I found a few possible issues with using the default solution and tried them one by one.

**1. reduce noise**

The noise that is added to the training was to much so i reduced the sigma from 0.2 to 0.1. This alone did not do a lot for the training results

![Changed sigma](https://github.com/fuzzballb/UdacityDDPGProject2/blob/master/images/Result1.PNG "Changed sigma")


**2. increase episode length**

Next I found that the agent probaly needed more time to get to it's goal then de maximum episode length that A specified. Thus the agent rarely got to it's goal. and if it did, it was because it was already close. This was the first time the score excided 1.0. Unfortunately it got stuck at around 2.x. not nearly the 30 i needed

![Changed episode length](https://github.com/fuzzballb/UdacityDDPGProject2/blob/master/images/Result2.PNG "Changed episode length")


**3. Normalize**

By defining batch normalisation and adding it to the forward pass

Still around the 2.x and not increasing.

![Normalisation](https://github.com/fuzzballb/UdacityDDPGProject2/blob/master/images/Result3.PNG "Normalisation")

**4. Increase replay buffer size**

Helped get above 3.x in the first step, but the learning didn't increase enough over time

![Increase replay buffer size](https://github.com/fuzzballb/UdacityDDPGProject2/blob/master/images/Result5.PNG "Increase replay buffer size")

**5. resetting the agent after every**

Agent.reset() helped get above 3.x in the first 100 episodes even without the previous 4. increased buffer size step
 
*No mayor change in learning*
 
**6. learninig for more then 500 episodes**

When trying to learn for more then 500 episodes connection with the Udacity environment gets lost.  

![More then 500 episodes](https://github.com/fuzzballb/UdacityDDPGProject2/blob/master/images/Result4.PNG "More then 500 episodes")

*Reloading saved weights to continue learning didn't work, probably because I didn't save and reload the *target* networks of the actor and critic.*

**7. increasing learning rate** 

When the steps in learning are to small, it can take a long time before the optimal value is found, make it to big, and you will overshoot your optimal value.

![Learning rate](https://github.com/fuzzballb/UdacityDDPGProject2/blob/master/images/Result6.PNG "Learning rate")


## Learning Algorithm

**1. First we initialize the agent in the Navigation notebook**

*Navigation.ipynb*

```Python
# initialise an agent
agent = Agent(state_size=33, action_size=4, random_seed=2)
```

**2. This sets the state and action size for the agent and creates an Actor, and a Critic neural net, with corresponding target network.** 

*ddpg_agent.ipynb*

```Python
        # Actor Network (w/ Target Network)
        self.actor_local = Actor(state_size, action_size, random_seed).to(device)
        self.actor_target = Actor(state_size, action_size, random_seed).to(device)
        self.actor_optimizer = optim.Adam(self.actor_local.parameters(), lr=LR_ACTOR)

        # Critic Network (w/ Target Network)
        self.critic_local = Critic(state_size, action_size, random_seed).to(device)
        self.critic_target = Critic(state_size, action_size, random_seed).to(device)
        self.critic_optimizer = optim.Adam(self.critic_local.parameters(), lr=LR_CRITIC, weight_decay=WEIGHT_DECAY)
```

DDPG is a algorithm that requires a actor and critic network. Both of these network also have target networks, that get small updates from the local network. This way the values don't change to much over a short time, and learning becomes more stable.


**3. Adding noise to increase exploration instead of only exploiting the paths that have lead to success**

*ddpg_agent.ipynb*

```Python
        # Noise process
        self.noise = OUNoise(action_size, random_seed)
```

Here we define noise that will be added to the action that is obtained from the actor network.


**4. Replay buffer, to store past experiences**

*ddpg_agent.ipynb*

```Python
        # Replay memory
        self.memory = ReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE, random_seed)
```

Because DDPG is a off policy algorithm (see bottom of the page), just like Q learning. It can learn from past experiences that are stored in the replay buffer. 


**5. Learn**

The critic network takes actions and states and produces a Q value. This is compared to the actual value in the environment, the difference between expected Q and actual reward from the environment is used to calculate a loss, which it tries to minimize. When the critic starts giving estimates about the Q value given states and actions, the actor network can use these trained values, to train the best action for a given state. 

for a more detailed explanation see the video below 

[![Youtube lecture about DDPG](https://img.youtube.com/vi/_pbd6TCjmaw/0.jpg)](https://www.youtube.com/watch?v=_pbd6TCjmaw&t=454). 



First, we get the next states from the stored experiences 

```Python
        states, actions, rewards, next_states, dones = experiences
```

Using these next_states we ask the actor network to predict the next actions

![Actor network](https://github.com/fuzzballb/UdacityDDPGProject2/blob/master/images/Actor.png "Actor network")

 
```Python
        # ---------------------------- update critic ---------------------------- #
        # Get predicted next-state actions and Q values from target models
        actions_next = self.actor_target(next_states)
```

Then we use these next actions to get the predicted Q targets using the Critic network 

![Half Critic network](https://github.com/fuzzballb/UdacityDDPGProject2/blob/master/images/HalfCritic.png "Half Critic network")

```Python       
        Q_targets_next = self.critic_target(next_states, actions_next)
```


Using the rewards and the dones from the experiences that came out of the replay buffer, plus the next Q targets (Q_targets_next) we just obtained we can calculate the target Q values (Q_targets).

```Python       
        # Compute Q targets for current states (y_i)
        Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))
 ```       
 
Then we check if we get the same Q values,  if we just add the states and actions to the local Critic network. 

![Full Critic network](https://github.com/fuzzballb/UdacityDDPGProject2/blob/master/images/FullCritic.png "Full Critic network")

```Python        
        # Compute critic loss
        Q_expected = self.critic_local(states, actions)
```       
 
If there is a difference, we change the weights of the local critic network to give results that are closer to the calculated Q values (Q_targets)

  ```Python        
        critic_loss = F.mse_loss(Q_expected, Q_targets)
        # Minimize the loss
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
```
When the Critic network is getting better at producing the Q values (using given states and actions as input), because it has been training on values gained from experiences. It starts giving the right Q values, even if it wouldn’t have the recorded experiences anymore.

At this point we can use the Critic’s learned states, actions to Q values relationship to train the Actor. We use the states from the replay memory as input for the Actor network to find the best action
 
![Actor network](https://github.com/fuzzballb/UdacityDDPGProject2/blob/master/images/Actor.png "Actor network")
 
```Python
        # ---------------------------- update actor ---------------------------- #
        # Compute actor loss
        actions_pred = self.actor_local(states)
```

Then we check with the trained Critic network what the Q values are and take the mean. If these Q values are high, then the Actor network made a good prediction. 

Because in training we try to minimize the loss, we take the negative of the Q values, because less is better 

```Python
        actor_loss = -self.critic_local(states, actions_pred).mean()
        # Minimize the loss
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
```


## on policy vs off policy

The on-policy aggression is like SARSA, they assume that their experience comes from the agents himself, and they try to improve agents policy right down this online stop. So, the agent plays and then it improves and plays again. And what you want to do is you want to get optimal Strategies as quickly as possible. 

Off-policy algorithms like Q-learning, as an example, they have slightly relax situation. They don't assume that the sessions you're obtained, you're training on, are the ones that you're going to use when the agent is going to finally get kind of unchained and applied to the actual problem. Because of this, it can use data from an experience replay buffer 



## Model based vs model free 

Model-based reinforcement learning has an agent try to understand the world and create a model to represent it. Here the model is trying to capture 2 functions, the transition function from states $T$ and the reward function $R$. From this model, the agent has a reference and can plan accordingly.

However, it is not necessary to learn a model, and the agent can instead learn a policy directly using algorithms like Q-learning or policy gradient.

A simple check to see if an RL algorithm is model-based or model-free is:

If, after learning, the agent can make predictions about what the next state and reward will be before it takes each action, it's a model-based RL algorithm.
