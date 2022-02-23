from imports import *

from lammpsrun import lammpsrun

import params

class lammpsenv(gym.Env):
    def __init__(self):
        self.run = lammpsrun(cores=params.cores)
        ats = self.run.get_atoms()
        self.topSats = ats[np.where(ats[:,1]==params.atom_type)]
        self.NSats = self.topSats.shape[0]
        self.action_space = gym.spaces.Discrete(self.NSats) #374 S atoms have ids from 1-374
        
        # Define a 2-D observation space
        self.shape = params.shape
        self.observation_shape = (self.shape, self.shape, 3) #PIL get RGB image and greyscale
        self.observation_space = Box(low = np.zeros(self.observation_shape), 
                                            high = np.ones(self.observation_shape)*255,
                                            dtype = np.float16)
        
        # Create a canvas to render the environment images upon 
        self.state = np.zeros(self.observation_shape)
        
        self.vacs = 0 #Running counter of vacs
        self.Nvacs = params.Nvacs #How many vacs to create
        self.baselineE = params.Baseline_E #Baseline corresponding to high tol w iteration max
        self.scale = params.scale_E_reward
        self.reward_range = (-np.inf,np.inf)
    
    def reset (self): #Reset the env at end of episode
        self.vacs = 0
        self.reward = 0
        self.run.reset()
        self.state = self.state_gen()
        return self.state
        
    def state_gen(self): #Generates state from one-hot encoding of defect positions
        img = self.run.viz_atoms(self.shape)*255
        return Image.fromarray(np.uint8(img)).convert('RGB')
        #Needs to be a PIL 
        
    def step(self, action):
        action = action + 1 #CONVERT from 0 indexing to start from index of 1
        info = {}
        self.vacs += 1
        self.run.deleted_atoms.append(action) #If not end of episode append atom to be deleted
        self.state = self.state_gen()
        if (self.vacs>=self.Nvacs):
            self.run.delete_atoms(self.run.deleted_atoms) #Only delete after all vacs are chosen
            done = True
            self.run.lmp.command(params.min_cmd)
            # TODO: Add some dumping for viz
            pe_per_at = self.run.get_pe_atom()
            reward = (self.baselineE-pe_per_at)*self.scale
            print("Reward", reward)
        else:
            done = False
            reward = 0
        return self.state, reward, done, info
    
    
    def render(self, mode = "human"):
        assert mode in ["human", "rgb_array"], "Invalid mode, must be either \"human\" or \"rgb_array\""
        if mode == "human":
            cv2.imshow("Game", self.state)
            cv2.waitKey(10)

        elif mode == "rgb_array":
            return np.array(self.state)
    