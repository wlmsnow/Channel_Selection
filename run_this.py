from env import Env
from RL_brain import DeepQNetwork


def run():
    step = 0
    for episode in range(1):
        # initial observation
        observation = env.reset()


        while step< 2:
            # fresh env

            # RL choose action based on observation
            action = RL.choose_action(observation)
            if action == 0:
                action_ = "Channel_1"
            elif action == 1:
                action_ = "Channel_6"
            else:
                action_ = "Channel_11"
            print(action_)
            # RL take action and get next observation and reward
            observation_, reward = env.step(action_)

            RL.store_transition(observation, action, reward, observation_)

            if (step > 200) and (step % 5 == 0):
                RL.learn()

            # swap observation
            observation = observation_

            # break while loop when end of this episode
            step += 1

    # end of game

   # env.destroy()


if __name__ == "__main__":
    # maze game
    env = Env()
    RL = DeepQNetwork(env.n_actions, env.n_features,
                      learning_rate=0.01,
                      reward_decay=0.9,
                      e_greedy=0.9,
                      replace_target_iter=200,
                      memory_size=2000,
                      # output_graph=True
                      )
    print(env.n_actions)
    print(env.n_features)
    run()
    #RL.plot_cost()



