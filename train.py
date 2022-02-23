from imports import *
import params
import os,sys
#os.environ["CUDA_VISIBLE_DEVICES"] = "7"
from torchsummary import summary
from agent import DDQNAgent

from lammpsrun import lammpsrun
from lammpsenv import lammpsenv

use_cuda = torch.cuda.is_available()
if use_cuda: 
    print(f"Using CUDA: {use_cuda}")
    print()
else:
    print("You neeed CUDA. Sorry!")
    exit()


#Get a PIL image and then greyscale it
class GrayScaleObservation(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.observation_space = Box(low=0, high=255, shape=self.observation_space.shape[:2], dtype=np.uint8)

    def observation(self, observation):
        transform = transforms.Grayscale()
        return transform(torch.tensor(np.transpose(observation, (2, 0, 1)).copy(), dtype=torch.float))

# Run through dummy or real resize to move everything to Torch tensors
class ResizeObservation(gym.ObservationWrapper):
    def __init__(self, env, shape):
        super().__init__(env)
        self.shape = (shape, shape)
        obs_shape = (1,) + self.shape + self.observation_space.shape[2:]
#        print(obs_shape) #Adding a 1 at the front cause not framestacking
        self.observation_space = Box(low=0, high=255, shape=obs_shape, dtype=np.uint8)

    def observation(self, observation):
        transformations = transforms.Compose([transforms.Resize(self.shape), transforms.Normalize(0, 255)])        
        return transformations(observation).numpy()#.squeeze(0) 
        #Think unsqueeze is needed because no framestack and need to batch


#Init env
env = lammpsenv()
env = GrayScaleObservation(env)
env = ResizeObservation(env, shape=params.shape)


print ("Observation shape is", env.observation_space.shape)

# Test run and image
#Render maps vacs onto 0-255 while next_state keeps in range 0-1 because of wrappers
import time
t1 = time.time()
env.reset()
ids = [266-1,265-1,308-1,307-1,328-1,327-1,348-1,347-1,245-1,246-1] #Diagonal
#IDS CAME FROM OVITO so subtract 1
for i in ids:
#    print (i)
    next_state, reward, done, info = env.step(i)
    if done:
        env.reset()
fr = env.render(mode="rgb_array")
f, ax = plt.subplots(1,2, figsize=(20, 12))
im=ax[0].imshow(fr, origin='lower')
plt.colorbar(im, ax=ax[0], fraction=0.046, pad=0.04)
im=ax[1].imshow(next_state.__array__().squeeze().T, origin='lower')
plt.colorbar(im, ax=ax[1], fraction=0.046, pad=0.04)
plt.savefig("tst_img.png")
plt.close()
print(next_state.shape)

t2 = time.time()
print ("Time", t2-t1)


#Load checkpoint
episode = params.episode
max_episodes = params.max_episodes


#Log rewards and network weights
rewards_file = params.rewards_file
save_directory = params.save_directory
if not os.path.exists (os.getcwd() + '/' + save_directory):
    os.mkdir(os.getcwd() + '/' + save_directory)


#Initialize agent
agent = DDQNAgent(action_dim=env.action_space.n, obs_dim = env.observation_space.shape,
                  save_directory=save_directory, rewards_file=rewards_file)


#Summarize model
model = agent.net
print(summary(model,env.observation_space.shape)) 


if params.load_checkpoint is not None: #Start from checkpoint?
    agent.load_checkpoint(save_directory + "/" + params.load_checkpoint) #Load weights
    memory_file = save_directory + "/memory_%d.pkl" %episode #Load experience replay deque
    agent.memory = pickle.load( open( memory_file, "rb" ) )
#    agent.current_step = episode * env.Nvacs #Load number of steps


#Training loop
while episode<max_episodes:
    state = env.reset()
    cstep = 0 #Keep track of steps in this episode
    while True:
        action = agent.act(state, env) #CHANGE from base: pass env here for masking
        if cstep==0: 
            action = np.random.randint(agent.action_dim) 
            #Take first action randomly so start from different places
        cstep+=1
        next_state, reward, done, info = env.step(action)
        agent.remember(state, next_state, action, reward, done)
        agent.experience_replay(reward, env) #CHANGE from base: pass env here for masking
        state = next_state
        if done:
            cstep = 0
            episode += 1
            agent.log_episode()
            if episode % params.log_period == 0: #Is it time to log results?
                agent.log_period(episode=episode, epsilon=agent.exploration_rate, step=agent.current_step)
                if episode % params.checkpoint_period == 0: #Is it time to save the model file and optionally memory?
                    agent.save_checkpoint(episode, env)
                plt.imshow(state.__array__().squeeze().T, origin='lower')
                plt.savefig(save_directory + "/img%d.png" %episode)
                plt.close()
            break

