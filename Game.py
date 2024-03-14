import numpy as np

import torch

from colorama import Fore, Back, Style, init


class Game:
    def __init__(self):
        self.board = np.zeros((4, 4), dtype=np.float32)
        self.reset_board()

        init(autoreset=True)

        self.color_map = {
            0: Fore.WHITE,
            2: Fore.GREEN,
            4: Fore.BLUE,
            8: Fore.CYAN,
            16: Fore.RED,
            32: Fore.MAGENTA,
            64: Fore.YELLOW,
            128: Fore.BLUE,
            256: Fore.MAGENTA,
            512: Fore.CYAN,
            1024: Fore.RED,
            2048: Fore.YELLOW,
            # 更多的数值和颜色可以根据需要添加
        }

    def reset_board(self):
        self.board = np.zeros((4, 4))
        # 随机选择2-3个点
        num_points = np.random.randint(2, 4)
        for _ in range(num_points):
            # 随机选择一个空位置
            empty_positions = np.argwhere(self.board == 0)
            random_position = empty_positions[np.random.randint(
                len(empty_positions))]
            # 在该位置上放置一个2
            self.board[random_position[0], random_position[1]] = 2

    def _add_new_2(self):
        # 找到self.board中所有值为0的元素的位置
        zero_positions = np.where(self.board == 0)
        # zero_positions是一个元组，其中包含行索引和列索引的数组

        # 获取0值的位置数量
        num_zeros = len(zero_positions[0])

        # 如果没有位置是0，则不执行任何操作
        if num_zeros == 0:
            return

        # 随机决定是添加一个还是两个2
        num_to_add = np.random.randint(
            1, min(3, num_zeros + 1))  # 确保不超过剩余的0的数量

        # 从0的位置中随机选择num_to_add个位置
        selected_indices = np.random.choice(
            range(num_zeros), size=num_to_add, replace=False)

        # 在选定的位置设置值为2
        for index in selected_indices:
            row = zero_positions[0][index]
            col = zero_positions[1][index]
            self.board[row, col] = 2

    def move(self, action):
        initial_board = self.board.copy()
        if action == 'w':
            board = np.rot90(self.board, 1)
            board, merged_score = self._move_left(board)
            self.board = np.rot90(board, -1)
        elif action == 's':
            board = np.rot90(self.board, -1)
            board, merged_score = self._move_left(board)
            self.board = np.rot90(board, 1)
        elif action == 'a':
            self.board, merged_score = self._move_left(self.board)
        elif action == 'd':
            board = np.rot90(self.board, 2)
            board, merged_score = self._move_left(board)
            self.board = np.rot90(board, -2)
        else:
            raise ValueError(
                f"Action must in Action_lst:['w', 'a', 's', 'd'], but got {action}")

        vac_num = np.sum(self.board == 0)

        self._add_new_2()

        board_unchanged = np.array_equal(initial_board, self.board)

        return merged_score, vac_num, board_unchanged

    def _move_left(self, board):
        merged_score = 0
        for i in range(4):
            collapes, merged_score_row = self._collapse(board[i])
            board[i] = collapes
            merged_score += merged_score_row

        return board, merged_score

    def _collapse(self, row):
        non_zero = row[row != 0]
        if len(non_zero) < 1:
            return row, 0

        collapes = np.zeros(4)

        if len(non_zero) < 2:
            collapes[0] = non_zero[0]
            return collapes, 0

        merged_score = 0
        i = 0
        k = 0
        while i < len(non_zero):
            # 当前项是最后一项，或者当前项与下一项不相等
            if i == len(non_zero) - 1 or non_zero[i] != non_zero[i+1]:
                collapes[k] = non_zero[i]
            else:
                # 合并当前项和下一项
                collapes[k] = non_zero[i] * 2
                merged_score += non_zero[i] * 2
                i += 1  # 跳过下一项

            i += 1
            k += 1

        return collapes, merged_score

    def print_board(self):
        for row in self.board:
            for value in row:
                color = self.color_map.get(value, Fore.LIGHTWHITE_EX)  # 默认颜色
                print(color + f"{int(value)}".rjust(4), end=' ')
            print()
        print(f"当前得分：\033[91m{int(self.__score())}\033[0m")

    def __score(self):

        return np.sum(self.board)

    def is_game_over(self):
        # 首先检查是否有任何单元格的值为0
        if np.any(self.board == 0):
            return False

        # 检查所有相邻单元格
        for i in range(self.board.shape[0]):
            for j in range(self.board.shape[1]):
                # 检查右侧单元格
                if j + 1 < self.board.shape[1] and self.board[i][j] == self.board[i][j + 1]:
                    return False  # 找到相同的相邻值
                # 检查下方单元格
                if i + 1 < self.board.shape[0] and self.board[i][j] == self.board[i + 1][j]:
                    return False  # 找到相同的相邻值

        # 如果没有空单元格也没有相同的相邻单元格，游戏结束
        return True

    def play(self):
        self.print_board()
        while not self.is_game_over():
            action = input("请给出你的移动方向：")
            if action == '' or action == 'end':
                break
            try:
                self.move(action)
                self.print_board()
            except ValueError as e:
                print(e)
        print(f"游戏结束，您的得分是：{int(self.__score())}")

    def get_board_tensor(self):
        board_tensor = torch.from_numpy(self.board)
        # 增加一个批次维度，并返回
        return board_tensor.unsqueeze(0)

    def get_max_tile(self):

        return np.max(self.board)


if __name__ == "__main__":
    game = Game()
    game.play()
