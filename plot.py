from matplotlib import pyplot as plt
import numpy as np
import json

# 滑动平均函数
def moving_average(a, window_size):
    cumulative_sum = np.cumsum(np.insert(a, 0, 0)) 
    middle = (cumulative_sum[window_size:] - cumulative_sum[:-window_size]) / window_size
    r = np.arange(1, window_size-1, 2)
    begin = np.cumsum(a[:window_size-1])[::2] / r
    end = (np.cumsum(a[:-window_size:-1])[::2] / r)[::-1]
    return np.concatenate((begin, middle, end))

with open("./result.json", 'r', encoding='utf-8') as f:
    result = json.load(f)

# 对每个指标数据应用滑动平均
window_size = 23  # 滑动平均的窗口大小
smoothed_return = moving_average(np.array(result["return"]), window_size)
smoothed_max_tile = moving_average(np.array(result["max_tile"]), window_size)
smoothed_turns = moving_average(np.array(result["turns"]), window_size)
smoothed_board_score = moving_average(np.array(result["board_score"]), window_size)
smoothed_ac = moving_average(np.array(result["ac_ls"]), window_size)
smoothed_cr = moving_average(np.array(result["cr_ls"]), window_size)

# 创建一个2x2的子图布局
fig, axs = plt.subplots(3, 2, figsize=(14, 10))

# 使用滑动平均后的数据绘制每个指标
axs[0, 0].plot(smoothed_return, label='Returns')
axs[0, 0].set_title('Returns')
axs[0, 0].set_xlabel('Episode')
axs[0, 0].set_ylabel('Value')
axs[0, 0].legend()

axs[0, 1].plot(smoothed_max_tile, label='Max Tiles')
axs[0, 1].set_title('Max Tiles')
axs[0, 1].set_xlabel('Episode')
axs[0, 1].set_ylabel('Value')
axs[0, 1].legend()

axs[1, 0].plot(smoothed_turns, label='Turns')
axs[1, 0].set_title('Turns')
axs[1, 0].set_xlabel('Episode')
axs[1, 0].set_ylabel('Value')
axs[1, 0].legend()

axs[1, 1].plot(smoothed_board_score, label='Board Score')
axs[1, 1].set_title('Board Score')
axs[1, 1].set_xlabel('Episode')
axs[1, 1].set_ylabel('Value')
axs[1, 1].legend()

axs[2, 0].plot(smoothed_ac, label='actor loss')
axs[2, 0].set_title('actor loss')
axs[2, 0].set_xlabel('Episode')
axs[2, 0].set_ylabel('actor loss')
axs[2, 0].legend()


axs[2, 1].plot(smoothed_cr, label='critic loss')
axs[2, 1].set_title('critic loss')
axs[2, 1].set_xlabel('Episode')
axs[2, 1].set_ylabel('critic loss')
axs[2, 1].legend()

plt.tight_layout()

plt.savefig("result.png", dpi=200)

plt.show()
