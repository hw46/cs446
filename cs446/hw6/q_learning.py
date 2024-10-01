import random
import tqdm
import numpy as np
import matplotlib.pyplot as plt

import gym

# TODO: implement this class
class QAgent:
    def __init__(self, state_space, action_space, init_val=-10):
        self.q_table = np.full((state_space.n, action_space.n), init_val)
        self.action_space = action_space

    def act(self, state, epsilon, train=False, action_mask=None):
        """Choose an action based on epsilon-greedy policy."""
        if train:
            return self._act_train(state, epsilon, action_mask)
        else:
            return self._act_eval(state, action_mask)
    
    def _act_train(self, state, epsilon, action_mask=None):
        """Implement epsilon-greedy strategy for action selection in training."""
        if random.random() < epsilon:
            # Exploration: Choose a random valid action
            valid_actions = [action for action, valid in enumerate(action_mask) if valid]
            return random.choice(valid_actions) if valid_actions else self.action_space.sample()
        else:
            # Exploitation: Choose the best action based on Q-values
            valid_q_values = {action: self.q_table[state][action] for action in range(self.action_space.n) if action_mask and action_mask[action]}
            return max(valid_q_values, key=valid_q_values.get) if valid_q_values else np.argmax(self.q_table[state])

    def _act_eval(self, state, action_mask=None):
        """Action selection in evaluation always picks the best action based on Q-table."""
        if action_mask:
            valid_q_values = {action: self.q_table[state][action] for action in range(self.action_space.n) if action_mask[action]}
            return max(valid_q_values, key=valid_q_values.get) if valid_q_values else np.argmax(self.q_table[state])
        return np.argmax(self.q_table[state])

    def update(self, state, action, reward, next_state, alpha, gamma):
        """Update the Q-value for the current state and action."""
        best_next_action = np.argmax(self.q_table[next_state])
        td_target = reward + gamma * self.q_table[next_state][best_next_action]
        td_error = td_target - self.q_table[state][action]
        self.q_table[state][action] += alpha * td_error
        
# -----------------------------------------------------------------------------
# NOTE: you do not need to modify the 3 functions below...
#       though you should do so for debugging purposes
# -----------------------------------------------------------------------------

def train_agent(env : gym.Env, agent : QAgent, epochs=10000, alpha=0.1, gamma=0.9, epsilon=0.1, use_action_mask=False):
    all_ep_rewards = []
    for _ in tqdm.tqdm(range(epochs)):
        state, info = env.reset()
        done = False
        ep_reward = 0.0
        while not done:
            action_mask = info.get('action_mask', [True] * env.action_space.n)
            action = agent.act(state, epsilon, action_mask)
            next_state, reward, done, info = env.step(action)
            agent.update(state, action, reward, next_state, alpha, gamma, info.get('action_mask', [True] * env.action_space.n))
            state = next_state
            ep_reward += reward
        all_ep_rewards.append(ep_reward)
    
    plt.figure(figsize=(10, 5))
    plt.plot(np.convolve(all_ep_rewards, np.ones(100)/100, mode='valid'), linewidth=2)
    plt.title('Training Rewards Over Time with No-Op Avoidance')
    plt.xlabel('Epochs')
    plt.ylabel('Smoothed Episode Rewards')
    plt.savefig('train_rewards.png')
    plt.clf()
    print(f"Training finished with mean reward: {np.mean(all_ep_rewards)}")
    
    return all_ep_rewards

def eval_agent(env : gym.Env, agent : QAgent, epsilon=0.1):
    state, info = env.reset()
    all_rewards = 0.0
    all_frames = [env.render()] # NOTE: assumes 'ansi' render_mode
    done = False
    
    while not done:
        action = agent.act(state, epsilon, train=False)
        
        state, reward, term, trunc, _ = env.step(action) 
        done = term or trunc
        
        all_frames.append(env.render())

        all_rewards += reward

    print(f"Obtained total reward of {np.sum(all_rewards)} after {len(all_frames)} steps")

    for frame in all_frames:
        print(frame)

def q_learning(alpha=0.1, gamma=0.9, epsilon=0.1, init_val=0.0, use_action_mask=False, save_path="train_rewards.png"):
    # create the environment, set the render_mode
    env = gym.make("Taxi-v3", render_mode="ansi")
    # initialize our agent
    agent = QAgent(env.observation_space, env.action_space, init_val=init_val)
    # train
    all_ep_rewards = train_agent(env, agent, alpha=alpha, gamma=gamma, epsilon=epsilon, use_action_mask=use_action_mask)
    # create moving average plot
    N = 10
    mov_avg_ep_rewards = np.convolve(np.array(all_ep_rewards), np.ones(N) / N, mode='valid')
    plt.plot(mov_avg_ep_rewards[:1000])
    plt.xlabel("Epochs")
    plt.ylabel("Episode Returns")
    plt.savefig(save_path)
    plt.clf()

    eval_agent(env, agent, epsilon=epsilon)

def train_masked_random_agent(env, agent, epochs=10000):
    all_ep_rewards = []
    episode_reached_goal_rate = 0

    for _ in tqdm.tqdm(range(epochs)):
        state, info = env.reset()
        done = False
        ep_reward = 0.0

        while not done:
            action_mask = info.get('action_mask', [True] * agent.action_space.n)  # Assume all actions are valid if not provided
            action = agent.act(state, action_mask)
            next_state, reward, done, info = env.step(action)
            ep_reward += reward
        
        all_ep_rewards.append(ep_reward)
        if reward > 0:  # Assuming reward > 0 means successful drop-off
            episode_reached_goal_rate += 1
    
    print("Training finished.\n")
    print("Episode success rate: {:.2f}%".format(100 * episode_reached_goal_rate / epochs))
    
    return all_ep_rewards

def plot_rewards(all_ep_rewards, save_path):
    plt.plot(all_ep_rewards)
    plt.title('Training Rewards Over Time')
    plt.xlabel('Epochs')
    plt.ylabel('Episode Rewards')
    plt.savefig(save_path)
    plt.clf()


if __name__ == "__main__":
    # We've filled in the experiment you should run for part b
    '''env = gym.make("Taxi-v3", render_mode="ansi")
    q_learning(alpha=0.1, gamma=0.9, epsilon=0.1, init_val=0, use_action_mask=False, save_path="train_rewards.png")'''
    '''experiments = {
        "5b": {
            "alpha": 0.1,
            "gamma": 0.9,
            "epsilon": 0.1, 
            "init_val": 0,
            "use_action_mask": False
        }
    }
    for exp_name in experiments:
        q_learning(**experiments[exp_name], save_path=f"train_rewards_{exp_name}.png")
        print("xxx")'''
        
        
    env = gym.make("Taxi-v3")
    agent = QAgent(env.observation_space, env.action_space, init_val=-10)
    train_agent(env, agent)