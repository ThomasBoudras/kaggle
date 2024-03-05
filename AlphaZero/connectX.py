import numpy as np
from collections import namedtuple

WinState = namedtuple('WinState', 'is_ended winner')
DEFAULT_HEIGHT = 6
DEFAULT_WIDTH = 7
DEFAULT_WIN_LENGTH = 4

class Board():
    """ConnectX Board (grille)"""

    def __init__(self,
                 height=None,
                 width=None,
                 win_length=None,
                 configuration=None):
        self.height = height or DEFAULT_HEIGHT
        self.width = width or DEFAULT_WIDTH
        self.win_length = win_length or DEFAULT_WIN_LENGTH
        
        if configuration is None:
            self.configuration = np.zeros([self.height, self.width], dtype=np.int32)
        else :
            self.configuration = configuration 
            assert self.configuration.shape == (self.height, self.width)

    def with_configuration(self, configuration):
        """update the board with the specific configuration"""
        return Board(self.height, self.width, self.win_length, configuration)

    def add_piece(self, column, player):
        """update the board when player chooses to add a piece in the column"""
        new_position, = np.where(self.configuration[:, column] == 0)
        # print('add piece', column)
        if len(new_position) == 0:
            print(self.configuration, column, player)
            raise ValueError( "Can't play column %s on the grid" % (column))

        self.configuration[new_position[-1]][column] = player
    
    def get_valid_moves(self):
        """Any zero value at the top row of the board is a valid move"""
        return self.configuration[0] == 0
    
    def _is_straight_winner(self, player_pieces):
        """check if player pieces has a horizontal win"""
        run_lengths = [player_pieces[:, i:i + self.win_length].sum(axis=1)
                       for i in range(len(player_pieces) - self.win_length  + 2)]
        return max([x.max() for x in run_lengths]) >= self.win_length

    def _is_diagonal_winner(self, player_pieces):
        """check if player pieces has a diagonal win"""
        win_length = self.win_length
        for i in range(len(player_pieces) - win_length + 1):
            for j in range(len(player_pieces[0]) - win_length + 1):
                if all(player_pieces[i + x][j + x] for x in range(win_length)):
                    return True
            for j in range(win_length - 1, len(player_pieces[0])):
                if all(player_pieces[i + x][j - x] for x in range(win_length)):
                    return True
        return False
    
    def get_winner(self):
        for player in [-1,1]:
            player_pieces = self.configuration == -player
            # print('get winner, player pieces and player:', player, player_pieces)

            if (self._is_straight_winner(player_pieces) 
                or self._is_straight_winner(player_pieces.transpose())
                or self._is_diagonal_winner(player_pieces)):
                return WinState(True, -player)
            
            if not self.get_valid_moves().any():
                return WinState(True, None) #there is no more move possible
            
        return WinState(False, None)
    
    def __str__(self):
        return str(self.configuration)

    

class Connect4Game(object):
    """
    Connect4 Game class implementing the alpha-zero-general Game interface.

    Use 1 for player1 and -1 for player2.
    """

    def __init__(self,
                 height=None,
                 width=None,
                 win_length=None,
                 configuration=None):
        self._base_board = Board(height, width, win_length, configuration)

    def getInitBoard(self):
        """
        Returns:
            startBoard: a representation of the board (ideally this is the form
                        that will be the input to your neural network)
        """
        return self._base_board.configuration

    def getBoardSize(self):
        """
        Returns:
            (x,y): a tuple of board dimensions
        """
        return (self._base_board.height, self._base_board.width)

    def getActionSize(self):
        """
        Returns:
            actionSize: number of all possible actions
        """
        return self._base_board.width

    def getNextState(self, board, player, action):
        """Returns a copy of the board with updated move, original board is unmodified.

        Input:
            board: current board
            player: current player (1 or -1)
            action: action taken by current player

        Returns:
            nextBoard: board after applying action
            nextPlayer: player who plays in the next turn (should be -player)

        """
        b = self._base_board.with_configuration(configuration=np.copy(board))
        # print('getnextstate', action)
        b.add_piece(action, player)
        return b.configuration, -player

    def getValidMoves(self, board, player):
        """Any zero value in top row in a valid move.

        Input:
            board: current board
            player: current player

        Returns:
            validMoves: a binary vector of length self.getActionSize(), 1 for
                        moves that are valid from the current board and player,
                        0 for invalid moves
        """
        return self._base_board.with_configuration(
            configuration=board).get_valid_moves()

    def getGameEnded(self, board, player):
        """
        Input:
            board: current board
            player: current player (1 or -1)

        Returns:
            r: 0 if game has not ended. 1 if player won, -1 if player lost,
               small non-zero value for draw.
               
        """
        b = self._base_board.with_configuration(configuration=board)
        winstate = b.get_winner()
        # print('getGameEnded, le board:', b)
        # print('getGameEnded, le winstate:', winstate)
        if winstate.is_ended:
            if winstate.winner is None:
                # draw has very little value.
                return 1e-4
            elif winstate.winner == player:
                return +1
            elif winstate.winner == -player:
                return -1
            else:
                raise ValueError('Unexpected winstate found: ', winstate)
        else:
            # 0 used to represent unfinished game.
            return 0

    def getCanonicalForm(self, board, player):
        """ 
        Input:
            board: current board
            player: current player (1 or -1)

        Returns:
            canonicalBoard: returns canonical form of board. The canonical form
                            should be independent of player. For e.g. in chess,
                            the canonical form can be chosen to be from the pov
                            of white. When the player is white, we can return
                            board as is. When the player is black, we can invert
                            the colors and return the board.
        """
        return board * player

    def getSymmetries(self, board, pi):
        """Board is left/right board symmetric

        Input:
            board: current board
            pi: policy vector of size self.getActionSize()

        Returns:
            symmForms: a list of [(board,pi)] where each tuple is a symmetrical
                       form of the board and the corresponding pi vector. This
                       is used when training the neural network from examples.
        """
        return [(board, pi),
                (np.array(board[:, ::-1], copy=True),
                 np.array(pi[::-1], copy=True))]

    def stringRepresentation(self, board):
        """
        Input:
            board: current board

        Returns:
            boardString: a quick conversion of board to a string format.
                         Required by MCTS for hashing.
        """
        return board.tostring()

    @staticmethod
    def display(board):
        print(" -----------------------")
        print(' '.join(map(str, range(len(board[0])))))
        print(board)
        print(" -----------------------")