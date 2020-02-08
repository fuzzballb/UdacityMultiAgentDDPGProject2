# UdacityMultiAgentDDPGProject3
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

![Train-two-individual-agents](https://github.com/fuzzballb/UdacityMultiAgentDDPGProject3/blob/master/images/Train-two-individual-agents.gif "Train-two-individual-agents")


## Solutions for getting a better score

**1. agents share a replay buffer**

Just like in real life, if you start playing tennis, the first thing you do is not trying to defeat the other player, but keep the ball in the air instead. You learn from your own actions, but you also learn from what the other player is doing. Together you form a collective memory of actions, given at states and their rewards. 

To implement this, the critic network and replay buffer needed to be seperated from the agents. Now the critic can learn from the experences of both agents at the same time. While the critic is learning the best Q value, given a state and action, the actors are learning from the critic individually.

![Shared-memory](https://github.com/fuzzballb/UdacityMultiAgentDDPGProject3/blob/master/images/Shared-memory.gif "Shared-memory")

**2. changing hyper parameters**

Different environments, need different hyper parameters. The parameters from the last project, didn't perform in our new multi agent environment, and there we had the most succes when editing the learning rate, of both the Actor and Critic. 

```python
LR_ACTOR = 1e-4         # learning rate of the actor 
LR_CRITIC = 3e-4        # learning rate of the critic
```

Changing the speed at which the target network is updated with the latest local network wights also had a positive effect

```python
TAU = 2e-1              # for soft update of target parameters
```

![Changed-hyper-parameters](https://github.com/fuzzballb/UdacityMultiAgentDDPGProject3/blob/master/images/Changed-hyper-parameters.gif "Changed-hyper-parameters")

**3. changing exploration noise **

While changing the noise function i first reset the sigma to 0.2 instead of 0.1. This alone did not do a lot for the training results. Changing  

```python
np.array([random.random() for i in range(len(x))])
```

to a random standaard deviation 

```python
np.random.standard_normal(self.size)
```

Did have a good effect on training source : https://github.com/agakshat/maddpg/blob/master/ExplorationNoise.py

![after-changing-noise-fuction](https://github.com/fuzzballb/UdacityMultiAgentDDPGProject3/blob/master/images/after-changing-noise-fuctio.gif "after-changing-noise-fuction")


## Learning Algorithm

**1. First we initialize two agents and an academy in the Navigation notebook**

*MADDPG.ipynb*

```Python
# initialise an agent
# initialise an agent
agent1 = Agent(device=DEVICE, key=0, state_size=24, action_size=2, random_seed=2, memory=shared_memory, noise=noise, checkpoint_folder=CHECKPOINT_FOLDER)
agent2 = Agent(device=DEVICE, key=1, state_size=24, action_size=2, random_seed=2, memory=shared_memory, noise=noise, checkpoint_folder=CHECKPOINT_FOLDER)
               
# initialise an academy
academy = Academy(device=DEVICE, state_size=24, action_size=2, random_seed=2, memory=shared_memory, checkpoint_folder=CHECKPOINT_FOLDER)
```

**2. This sets the state and action size for the agent and creates an Actor, and a Critic neural net, with corresponding target network.** 

*ddpg_agent.py*

In the Agent class (actor)

```Python
        # Actor Network (w/ Target Network)
        self.actor_local = Actor(state_size, action_size, random_seed).to(self.device)
        self.actor_target = Actor(state_size, action_size, random_seed).to(self.device)
        self.actor_optimizer = optim.Adam(self.actor_local.parameters(), lr=LR_ACTOR)
```

In the Academy class (critic)

```Python
        # Critic Network (w/ Target Network)
        self.critic_local = Critic(state_size, action_size, random_seed).to(self.device)
        self.critic_target = Critic(state_size, action_size, random_seed).to(self.device)
        self.critic_optimizer = optim.Adam(self.critic_local.parameters(), lr=LR_CRITIC, weight_decay=WEIGHT_DECAY)
```   

DDPG is a algorithm that requires a actor and critic network. Both of these network also have target networks, that get small updates from the local network. The weights of the local network are slowly copied to the target networks for the learning to become more stable. The copyspeed is determined by the TAU hyper parameter

```Python
        # ----------------------- update target networks ----------------------- #
        self.soft_update(self.critic_local, self.critic_target, TAU)
        self.soft_update(actor.actor_local, actor.actor_target, TAU)
```           

**3. Adding noise to increase exploration instead of only exploiting the paths that have lead to success**

*MADDPG.ipynb*

```Python
        noise = OUNoise(action_size, 2)
```

Here we define noise that will be added to the action that is obtained from the actor network. Both of the actors need to train in the same environment, thus the same noise also needs to be shared by both agents.

```Python
# initialise an agent
agent1 = Agent(device=DEVICE, key=0, state_size=24, action_size=2, random_seed=2, memory=shared_memory, noise=noise, checkpoint_folder=CHECKPOINT_FOLDER)
agent2 = Agent(device=DEVICE, key=1, state_size=24, action_size=2, random_seed=2, memory=shared_memory, noise=noise, checkpoint_folder=CHECKPOINT_FOLDER)
```


**4. Replay buffer, to store past experiences**

*MADDPG.ipynb*

```Python
        shared_memory = ReplayBuffer(DEVICE, action_size, BUFFER_SIZE, BATCH_SIZE, 2)
```

Because DDPG is a off policy algorithm, just like Q learning. It can learn from past experiences that are stored in the replay buffer. As discust before, the replay buffer of both Agents need to be the same replay buffer, so the critic can learn from both agents at the same time.

*MADDPG.ipynb*

```Python
        # initialise an agent
        agent1 = Agent(device=DEVICE, key=0, state_size=24, action_size=2, random_seed=2, memory=shared_memory, noise=noise, checkpoint_folder=CHECKPOINT_FOLDER)
        agent2 = Agent(device=DEVICE, key=1, state_size=24, action_size=2, random_seed=2, memory=shared_memory, noise=noise, checkpoint_folder=CHECKPOINT_FOLDER)

        # initialise an academy
        academy = Academy(device=DEVICE, state_size=24, action_size=2, random_seed=2, memory=shared_memory, checkpoint_folder=CHECKPOINT_FOLDER)
```


**5. Learn**

The critic network takes actions and states and produces a Q value. This is compared to the actual value in the environment, the difference between expected Q and actual reward from the environment is used to calculate a loss, which it tries to minimize. When the critic starts giving estimates about the Q value given states and actions, both actor networks can use these trained values, to train the best action for a given state. 

for a more detailed explanation of DDPG actor critc, see the video below 

[![Youtube lecture about DDPG](https://img.youtube.com/vi/_pbd6TCjmaw/0.jpg)](https://www.youtube.com/watch?v=_pbd6TCjmaw&t=454). 


*ddpg_agent.py*

First, we get the next states from the stored experiences and feed these to the Academy (Critic)


```Python
class Academy:    
   ...
   def learn(self, actor, experiences, gamma):
   ...
        states, actions, rewards, next_states, dones = experiences
```

Using these next_states we ask the provided Actor network to predict the next actions

![Actor network](https://github.com/fuzzballb/UdacityDDPGProject2/blob/master/images/Actor.png "Actor network")

 
```Python
        # ---------------------------- update critic ---------------------------- #
        # Get predicted next-state actions and Q values from target models
        actions_next = actor.actor_target(next_states)
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

At this point we can use the Critic’s learned states, actions to Q values relationship to train the provided Actor. We use the states from the shared replay memory as input for the provided Actor network to find the best action
 
![Actor network](https://github.com/fuzzballb/UdacityDDPGProject2/blob/master/images/Actor.png "Actor network")
 
```Python
        # ---------------------------- update actor ---------------------------- #
        # Compute actor loss
        actions_pred = actor.actor_local(states)
```

Then we check with the trained Critic network what the Q values are and take the mean. If these Q values are high, then the Actor network made a good prediction. 

Because in training we try to minimize the loss, we take the negative of the Q values (less is better)  

```Python
        actor_loss = -self.critic_local(states, actions_pred).mean()
        # Minimize the loss
        actor.actor_optimizer.zero_grad()
        actor_loss.backward()
        actor.actor_optimizer.step()
```


