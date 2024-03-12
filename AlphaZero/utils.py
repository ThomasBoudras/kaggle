#These codes were inspired and modified by the following two github sites: "https://github.com/PaddlePaddle/PARL/tree/develop" and "https://github.com/suragnair/alpha-zero-general/tree/master".

class AverageMeter(object):
    """From https://github.com/pytorch/examples/blob/master/imagenet/main.py"""

    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def __repr__(self):
        return f'{self.avg:.2e}'

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class dotdict(dict):
    def __getattr__(self, name):
        return self[name]


def win_loss_draw(score):
    if score > 0:
        return 'win'
    if score < 0:
        return 'loss'
    return 'draw'


split_group = lambda the_list, group_size: zip(*(iter(the_list), ) * group_size)

import numpy as np
import json
from simulation_game import Simulation_Connect4


def get_test_dataset():
    game = Simulation_Connect4()
    test_dataset = []
    with open("refmoves1k_kaggle") as f:
        for line in f:
            data = json.loads(line)

            board = data["board"]
            board = np.reshape(board, game.getBoardSize()).astype(int)
            board[np.where(board == 2)] = -1

            # find out how many moves are played to set the correct mark.
            ply = len([x for x in data["board"] if x > 0])
            if ply & 1:
                player = -1
            else:
                player = 1

            test_dataset.append({
                'board': board,
                'player': player,
                'move_score': data['move score'],
            })
    return test_dataset