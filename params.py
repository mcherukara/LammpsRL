#Checkpoint params
checkpoint_period = 1000 #How often to save model
log_period = 10 #How often to output model performance
episode = 0
max_episodes = 10000
if episode:
    load_checkpoint = "checkpoint_%d.pth" %episode
else:
    load_checkpoint = None


#Log rewards and network weights
rewards_file = 'MoS2/rewards2.txt'
save_directory = "MoS2/v2"
save_memory = False #Save memory on checkpoint yes/no


#DQNAgent params
exploration_rate_decay = 0.999995
exploration_rate_min = 0.01
learn_every = 4 #How many env steps to train
exploration_rate_start = 1.0 #Initial exploration rate
gamma = 0.95 #Reward decay
memory_size = 100000


#Training params
batch_size = 32
learning_rate = 1e-4


#Misc
seed = 1 #Random seed for reproduce
model_type = "Duel_Double_DQN" #Duel_Double_DQN or Double_DQN


#MD specific
Nvacs = 20 #How many vacancies in a step
Baseline_E = -5.029576 #Average E of random arrangements of 40 vacs
E_conversion = 4.3363*10**-2 #kcal/mol to eV for ReaxFF runs
cores = 64 #How many cores to use?
in_file = "MoS2/in.MoS2"
scale_E_reward = 1e4 #Scale E diff to get a reasonable number
min_cmd = "minimize 1.0e-14 1.0e-16 60000 120000" #Minimize to be run to get PE
shape = 36 #Shape of 2D image
atom_type = 3 #Type of atom to delete