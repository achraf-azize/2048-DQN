from game_display import *

if __name__ == '__main__':

    reward = 2   # Change here
    reward_mode = 'scores' * int(reward == 1) + 'nb_merge_max_tile' * (int(reward == 2)) + 'nb_empty_tiles' * int(reward == 3)
    test_env = Env(reward_mode=reward_mode)
    filename = 'parameters_reward_' + str(reward) + '.dms'


    QNetwork = DQN()
    QNetwork.load_state_dict(torch.load(filename, map_location={'cuda:0': 'cpu'}))

    play_game_fancy(test_env, QNetwork)



