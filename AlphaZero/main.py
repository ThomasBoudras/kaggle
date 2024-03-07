from Coach import Coach
from connect4_game import Connect4Game
from alphazero_agent import AlphaZeroAgent as NN
from utils import *

import logging
log = logging.getLogger(__name__)


args = dotdict({
    'numIters': 1000,
    'numEps': 5,              # Number of complete self-play games to simulate during a new iteration. default=100
    'tempThreshold': 15,        #
    'updateThreshold': 0.6,     # During arena playoff, new neural net will be accepted if threshold or more of games are won.
    'maxlenOfQueue': 200000,    # Number of game examples to train the neural networks.
    'numMCTSSims': 25,          # Number of games moves for MCTS to simulate.
    'arenaCompare': 40,         # Number of games to play during arena play to determine if new net will be accepted.
    'cpuct': 1,

    'checkpoint': './temp/',
    'load_model': True,
    'load_folder_file': ('./temp/','best.pth.tar'),
    'numItersForTrainExamplesHistory': 20,
})


def main():
    log.info('Loading %s...', Connect4Game.__name__)
    game = Connect4Game()

    log.info('Loading %s...', NN.__name__)
    model = NN(game)

    if args.load_model:
        log.info('Loading checkpoint {}...'.format(args.load_folder_file))
        model.load_checkpoint(args.load_folder_file[0], args.load_folder_file[1])
    else:
        log.warning('Not loading a checkpoint!')
    
    log.info('Loading the Coach...')
    c = Coach(game, model, args)
    
    if args.load_model:
        log.info('Loading coach')
        c.loadTrainExamples()

    c.learn()


if __name__ == "__main__":
    main()
