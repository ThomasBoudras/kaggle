import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Connect4Model(nn.Module):
    def __init__(self, game, args):
        self.board_x, self.board_y = game.getBoardSize()
        self.action_size = game.getActionSize()
        self.args = args

        super(Connect4Model, self).__init__()
        self.conv1 = nn.Conv2d(1, args.num_channels, 3, stride=1, padding=1).to(device)
        self.conv2 = nn.Conv2d(
            args.num_channels, args.num_channels, 3, stride=1, padding=1).to(device)
        self.conv3 = nn.Conv2d(
            args.num_channels, args.num_channels, 3, stride=1).to(device)
        self.conv4 = nn.Conv2d(
            args.num_channels, args.num_channels, 3, stride=1).to(device)

        self.bn1 = nn.BatchNorm2d(args.num_channels).to(device)
        self.bn2 = nn.BatchNorm2d(args.num_channels).to(device)
        self.bn3 = nn.BatchNorm2d(args.num_channels).to(device)
        self.bn4 = nn.BatchNorm2d(args.num_channels).to(device)

        self.fc1 = nn.Linear(
            args.num_channels * (self.board_x - 4) * (self.board_y - 4), 128).to(device)
        self.fc_bn1 = nn.BatchNorm1d(128).to(device)

        self.fc2 = nn.Linear(128, 64).to(device)
        self.fc_bn2 = nn.BatchNorm1d(64).to(device)

        self.fc3 = nn.Linear(64, self.action_size).to(device)

        self.fc4 = nn.Linear(64, 1).to(device)
    
    def forward(self, s):
        """
        Args:
            s(torch.Tensor): batch_size x board_x x board_y
        """
        # batch_size x 1 x board_x x board_y
        s = s.view(-1, 1, self.board_x, self.board_y)
        # batch_size x num_channels x board_x x board_y
        s = F.relu(self.bn1(self.conv1(s)))
        # batch_size x num_channels x board_x x board_y
        s = F.relu(self.bn2(self.conv2(s)))
        # batch_size x num_channels x (board_x-2) x (board_y-2)
        s = F.relu(self.bn3(self.conv3(s)))
        # batch_size x num_channels x (board_x-4) x (board_y-4)
        s = F.relu(self.bn4(self.conv4(s)))
        s = s.view(
            -1,
            self.args.num_channels * (self.board_x - 4) * (self.board_y - 4))

        s = F.dropout(
            F.relu(self.fc_bn1(self.fc1(s))),
            p=self.args.dropout,
            training=self.training)  # batch_size x 128
        s = F.dropout(
            F.relu(self.fc_bn2(self.fc2(s))),
            p=self.args.dropout,
            training=self.training)  # batch_size x 64

        pi = self.fc3(s)  # batch_size x action_size
        v = self.fc4(s)  # batch_size x 1

        return F.log_softmax(pi, dim=1), torch.tanh(v)