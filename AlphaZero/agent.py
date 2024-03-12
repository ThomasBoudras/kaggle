#These codes were inspired and modified by the following two github sites: "https://github.com/PaddlePaddle/PARL/tree/develop" and "https://github.com/suragnair/alpha-zero-general/tree/master".


import os
import numpy as np
import torch
import torch.optim as optim

from tqdm import tqdm
from utils import *
from model_conv import model_NN

args = dotdict({
    'lr': 0.001,
    'dropout': 0.3,
    'epochs': 5,
    'batch_size': 64,
    'num_channels': 64,
})


class alpha_zero():
    def __init__(self, model):
        self.model = model

    def learn(self, boards, target_pis, target_vs, optimizer):
        self.model.train()  # train mode

        # compute model output
        out_log_pi, out_v = self.model(boards)

        pi_loss = -torch.sum(target_pis * out_log_pi) / target_pis.size()[0]

        v_loss = torch.sum((target_vs - out_v.view(-1))**2) / target_vs.size()[0]

        total_loss = pi_loss + v_loss

        # compute gradient and do SGD step
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        return total_loss, pi_loss, v_loss

    def predict(self, board):
        self.model.eval()  # eval mode

        with torch.no_grad():
            log_pi, v = self.model(board)

        pi = torch.exp(log_pi)
        return pi, v


class agent_alpha_zero():
    def __init__(self, game):
        super(agent_alpha_zero, self).__init__()
        self.model = model_NN(game, args)
        self.alg = alpha_zero(self.model)
        self.cuda = torch.cuda.is_available()
        self.board_x, self.board_y = game.getBoardSize()
        self.action_size = game.getActionSize()

    def learn(self, examples):
        """
        Args:
            examples: list of examples, each example is of form (board, pi, v)
        """
        optimizer = optim.Adam(self.alg.model.parameters(), lr=args.lr)

        for epoch in range(args.epochs):
            print('EPOCH ::: ' + str(epoch + 1))

            batch_count = int(len(examples) / args.batch_size)

            pbar = tqdm(range(batch_count), desc='Training Net')
            for _ in pbar:
                sample_ids = np.random.randint(
                    len(examples), size=args.batch_size)
                boards, pis, vs = list(zip(*[examples[i] for i in sample_ids]))
                boards = torch.FloatTensor(np.array(boards).astype(np.float64))
                target_pis = torch.FloatTensor(np.array(pis))
                target_vs = torch.FloatTensor(np.array(vs).astype(np.float64))

                if self.cuda:
                    boards, target_pis, target_vs = boards.contiguous().cuda(), target_pis.contiguous().cuda(), target_vs.contiguous().cuda()

                total_loss, pi_loss, v_loss = self.alg.learn( boards, target_pis, target_vs, optimizer)

                # record loss with tqdm
                pbar.set_postfix(Loss_pi=pi_loss.item(), Loss_v=v_loss.item())

    def predict(self, board):
        """
        Args:
            board (np.array): input board

        Return:
            pi (np.array): probability of actions
            v (np.array): estimated value of input
        """
        # preparing input
        board = torch.FloatTensor(board.astype(np.float64))
        if self.cuda:
            board = board.contiguous().cuda()
        board = board.view(1, self.board_x, self.board_y)

        pi, v = self.alg.predict(board)

        return pi.data.cpu().numpy()[0], v.data.cpu().numpy()[0]


    def save_checkpoint(self, folder='checkpoint', filename='checkpoint.pth.tar'):
        filepath = os.path.join(folder, filename)
        if not os.path.exists(folder):
            print("Checkpoint Directory does not exist! Making directory {}".format(folder))
            os.mkdir(folder)
        else:
            print("Checkpoint Directory exists! ")
        torch.save({
            'state_dict': self.model.state_dict(),
        }, filepath)

    def load_checkpoint(self, folder='checkpoint', filename='checkpoint.pth.tar'):
        filepath = os.path.join(folder, filename)
        if not os.path.exists(filepath):
            raise ("No model in path {}".format(filepath))
        map_location = None if self.cuda else 'cpu'
        checkpoint = torch.load(filepath, map_location=map_location)
        self.model.load_state_dict(checkpoint['state_dict'])
