#These codes were inspired and modified by the following two github sites: "https://github.com/PaddlePaddle/PARL/tree/develop" and "https://github.com/suragnair/alpha-zero-general/tree/master".

from training import Train
from simulation_game import Simulation_Connect4
from agent import agent_alpha_zero as agent

import logging
log = logging.getLogger(__name__)

class dotdict(dict):
    def __getattr__(self, name):
        return self[name]

args = dotdict({
    'numIters': 1000,
    'numEps': 2,              # Number of complete self-play games to simulate during a new iteration. default=100
    'tempThreshold': 15,        #
    'updateThreshold': 0.6,     # During arena playoff, new neural net will be accepted if threshold or more of games are won.
    'maxlenOfQueue': 200000,    # Number of game examples to train the neural networks.
    'numMCTSSims': 25,          # Number of games moves for MCTS to simulate.
    'fightCompare': 40,         # Number of games to play during arena play to determine if new net will be accepted.
    'cpuct': 1,

    'checkpoint': './models/',
    'load_model': True,
    'load_folder_file': ('./models/','best.pth.tar'),
    'numItersForTrainExamplesHistory': 20,
})


def main():
    game = Simulation_Connect4()

    log.info('Initialisation')
    model = agent(game)

    if args.load_model:
        model.load_checkpoint(args.load_folder_file[0], args.load_folder_file[1])
    
    log.info('Training')
    c = Train(game, model, args)
    
    if args.load_model:
        c.loadTrainExamples()

    c.learn()


if __name__ == "__main__":
    main()
