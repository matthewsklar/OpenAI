import gym

from CartPole.agent import Agent

if __name__ == '__main__':
    env = gym.make("CartPole2-v0")
    agent = Agent(env.action_space)

    episode_count = 500

    for i in range(episode_count):
        state = env.reset()

        reward_sum = 0
        done = False

        while not done:
            env.render()

            action = agent.act(state)
            state, reward, done, info = env.step(action)

            reward_sum += 1

        if done:
            print('Episode %d: Ended with score %d' % (i + 1, reward_sum))

    env.close()