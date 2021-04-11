[//]: # (Image References)

[image1]: https://user-images.githubusercontent.com/10624937/42135619-d90f2f28-7d12-11e8-8823-82b970a54d7e.gif "Trained Agent"

# Learning from Pixels

### Introduction

We will train an agent to navigate (and collect bananas!) in a large, square world.  

![Trained Agent][image1]

A reward of +1 is provided for collecting a yellow banana, and a reward of -1 is provided for collecting a blue banana.  Thus, the goal of your agent is to collect as many yellow bananas as possible while avoiding blue bananas.  

The state is an 84 x 84 RGB image, corresponding to the agent's first-person view. Given this information, the agent has to learn how to best select actions. Four discrete actions are available, corresponding to:
- **`0`** - move forward.
- **`1`** - move backward.
- **`2`** - turn left.
- **`3`** - turn right.

The task is episodic, and in order to solve the environment, your agent must get an average score of +13 over 100 consecutive episodes.

### Getting Started

1. Install packages required to working the code:
    - Python 3.6
    - `pip install unityagents` Unity Machine Learning Agents (ML-Agents)
    - PyTorch (The code works with both CPU and GPU that is capable of running CUDA)
    - NumPy, Matplotlib, Pandas

2. Download the environment from one of the links below. You need only select the environment that matches your operating system:
    - Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/VisualBanana_Linux.zip)
    - Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/VisualBanana.app.zip)
    - Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/VisualBanana_Windows_x86.zip)
    - Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/VisualBanana_Windows_x86_64.zip)
    
    (_For Windows users_) Check out [this link](https://support.microsoft.com/en-us/help/827218/how-to-determine-whether-a-computer-is-running-a-32-bit-version-or-64) if you need help with determining if your computer is running a 32-bit version or 64-bit version of the Windows operating system.
    
3. Place the file in folder with other repository files, and unzip (or decompress) the file.

__*Note*:__ The project environment is similar to, but __not identical to the Banana Collector environment__ on the [Unity ML-Agents GitHub page](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Learning-Environment-Examples.md#banana-collector).

### Instructions

The repository contains four files and one directory:

- `double_dqn_agent.py`: DDQN agent with Experience replay 
- `model.py`: CNN model 
- `Navigation_Pixels.ipynb`: The code to explore the environment and train an agent
- `checkpoint`: Saved trained model weights of the successful agent as a multipart ZIP file
- `Report.md`: Description of implementation

Follow the instructions in `Navigation_Pixels.ipynb` to get started with training an agent.
