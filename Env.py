import numpy as np

from Game import Game


class Env(Game):
    def __init__(self, feature_dim):
        super().__init__()
        self.index_to_action = {
            0: 'w',
            1: 'a',
            2: 's',
            3: 'd'
        }
        self.time_step = 0
        self.feature_dim = feature_dim

    def encode_board(self):
        # 初始化一个形状为(feature_dim, 4, 4)的数组，用于存储编码后的棋盘
        encoded_board = np.zeros((self.feature_dim, 4, 4), dtype=np.int8)

        for i in range(4):  # 遍历棋盘的行
            for j in range(4):  # 遍历棋盘的列
                # 获取当前位置的数值并转换为二进制字符串，去掉开头的'0b'
                binary_value = bin(int(self.board[i, j]))[2:]
                # 计算需要填充的0的数量
                padding = self.feature_dim - len(binary_value)
                # 将二进制字符串转换为数字列表，并确保长度等于feature_dim
                binary_list = [0] * padding + \
                    [int(digit) for digit in binary_value]
                # 更新encoded_board
                for k in range(self.feature_dim):
                    encoded_board[k, i, j] = binary_list[k]

        return encoded_board

    def reset(self):
        self.reset_board()
        self.time_step = 0
        return self.encode_board()

    def step(self, action):
        self.time_step += 1
        action = self.index_to_action[action]
        merged_score, vac_num, board_unchanged = self.move(action)
        # print(merged_score)
        vac_score = np.sqrt(vac_num)
        max_tile_score = np.log2(self.get_max_tile())
        next_state = self.encode_board()
        done = self.is_game_over()
        # reward = vac_score + 2 * np.log2(np.sum(self.board)) + max_tile_score / 2 + merged_score / 2
        reward = np.log2(merged_score + 1) + vac_num
        # reward -= np.sqrt(np.abs(np.sum(self.board) / 3 - self.time_step) + 1)
        # print(reward)
        # reward = -1

        if board_unchanged:
            reward = -1

        if done:
            reward -= 50 * (8 - np.log2(self.get_max_tile()))

        max_tile = self.get_max_tile()
        board_score = np.sum(self.board)

        return next_state, reward, done, max_tile, board_score, self.encode_board()
