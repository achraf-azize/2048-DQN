import torch
import torch.nn as nn
import torch.nn.functional as F
import gym_2048
import time
import numpy as np
import math


class DQN(nn.Module):

    def __init__(self):
        super(DQN, self).__init__()

        self.conv_a = nn.Conv2d(16, 128, kernel_size=(1, 2))
        self.conv_b = nn.Conv2d(16, 128, kernel_size=(2, 1))

        self.conv_aa = nn.Conv2d(128, 128, kernel_size=(1, 2))
        self.conv_ab = nn.Conv2d(128, 128, kernel_size=(2, 1))

        self.conv_ba = nn.Conv2d(128, 128, kernel_size=(1, 2))
        self.conv_bb = nn.Conv2d(128, 128, kernel_size=(2, 1))

        self.fc = nn.Sequential(
            nn.Linear(7424, 256),
            nn.ReLU(),
            nn.Linear(256, 4)
        )

    def forward(self, x):
        x_a = F.relu(self.conv_a(x))
        x_b = F.relu(self.conv_b(x))

        x_aa = F.relu(self.conv_aa(x_a))
        x_ab = F.relu(self.conv_ab(x_a))

        x_ba = F.relu(self.conv_ba(x_b))
        x_bb = F.relu(self.conv_bb(x_b))

        sh_a = x_a.shape
        sh_aa = x_aa.shape
        sh_ab = x_ab.shape
        sh_b = x_b.shape
        sh_ba = x_ba.shape
        sh_bb = x_bb.shape

        x_a = x_a.view(sh_a[0], sh_a[1] * sh_a[2] * sh_a[3])
        x_aa = x_aa.view(sh_aa[0], sh_aa[1] * sh_aa[2] * sh_aa[3])
        x_ab = x_ab.view(sh_ab[0], sh_ab[1] * sh_ab[2] * sh_ab[3])
        x_b = x_b.view(sh_b[0], sh_b[1] * sh_b[2] * sh_b[3])
        x_ba = x_ba.view(sh_ba[0], sh_ba[1] * sh_ba[2] * sh_ba[3])
        x_bb = x_bb.view(sh_bb[0], sh_bb[1] * sh_bb[2] * sh_bb[3])

        concat = torch.cat((x_a, x_b, x_aa, x_ab, x_ba, x_bb), dim=1)

        output = self.fc(concat)

        return output

class Env(gym_2048.Base2048Env):

        metadata = {
            'render.modes': ['human'], 'rewards': ['score', 'nb_merge', 'nb_merge_max_tile']
        }

        def __init__(self, reward_mode='score'):
            super().__init__()
            self.reward_mode = reward_mode

        def step(self, action: int):

            # Align board action with left action
            rotated_obs = np.rot90(self.board, k=action)
            reward, updated_obs = self._slide_left_and_merge(rotated_obs)
            self.board = np.rot90(updated_obs, k=4 - action)

            # Place one random tile on empty location
            time.sleep(0.2)
            self._place_random_tiles(self.board, count=1)

            done = self.is_done()

            return self.board, reward, done, {}

        def _slide_left_and_merge(self, board):

            result = []

            score = 0
            max_tile = 0
            for row in board:
                row = np.extract(row > 0, row)
                score_, max_row, result_row = self._try_merge(row)
                max_tile = max(max_tile, max_row)
                score += score_
                row = np.pad(np.array(result_row), (0, 4 - len(result_row)),
                             'constant', constant_values=(0,))
                result.append(row)

            score = score + int(self.reward_mode == 'nb_merge_max_tile') * np.log2(max_tile)
            return score, np.array(result, dtype=np.int64)

        def _try_merge(self, row):
            score = 0
            result_row = []

            i = 1
            max_row = 0
            while i < len(row):
                if row[i] == row[i - 1]:
                    score += (row[i] + row[i - 1]) * int(self.reward_mode == 'score') + int(self.reward_mode != 'score')
                    max_row = max(max_row, row[i] + row[i - 1])
                    result_row.append(row[i] + row[i - 1])
                    i += 2
                else:
                    max_row = max(max_row, row[i - 1])
                    result_row.append(row[i - 1])
                    i += 1

            if i == len(row):
                max_row = max(max_row, row[i - 1])
                result_row.append(row[i - 1])

            return score, max_row, result_row


def change_values(X):
    power_mat = np.zeros(shape=(1,16,4,4),dtype=np.float32)
    for i in range(4):
        for j in range(4):
            if(X[i][j]==0):
                power_mat[0][0][i][j] = 1.0
            else:
                power = int(math.log(X[i][j],2))
                power_mat[0][power][i][j] = 1.0
    return power_mat

def testing_results(reward):
    reward_mode = 'scores' * int(reward == 1) + 'nb_merge_max_tile' * (int(reward == 2)) + 'nb_empty_tiles' * int(
        reward == 3)
    test_env = Env(reward_mode=reward_mode)

    QNetwork = DQN().to(device)
    filename = 'parameters_reward_' + str(reward)
    QNetwork.load_state_dict(torch.load(path + filename))

    dic_max = {}
    for i in range(13):
        dic_max[2 ** i] = 0

    n_games = 1000

    for k in range(n_games):
        if (k % 50 == 0):
            print(str(k) + " games played ")
        grid = play_game(test_env, QNetwork, render=False)
        max_tile = np.max(grid)
        dic_max[max_tile] = dic_max[max_tile] + 1

    for key in dic_max:
        dic_max[key] = dic_max[key] / n_games

    return (dic_max)