import gymnasium as gym
env = gym.make("Taxi-v3", render_mode="ansi")

self.state_space=env.observation_space
        self.action_space=env.action_space
        self.table=np.zeros([state_space,action_space])


if random.uniform(0, 1) < epsilon:
            action = env.action_space.sample()
        else:
            action = np.max(self.table)

        next_state, reward, _, _ = env.step(action)
        old_q_value = q_table[state][action]

        # Check if next_state has q values already
        if not q_table[next_state]:
            q_table[next_state] = {action: 0 for action in range(env.action_space.n)}

        # Maximum q_value for the actions in next state
        next_max = max(q_table[next_state].values())

        # Calculate the new q_value
        new_q_value = (1 - alpha) * old_q_value + alpha * (reward + gamma * next_max)

        # Finally, update the q_value
        q_table[state][action] = new_q_value

        return next_state, reward