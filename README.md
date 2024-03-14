### 基于PPO的2048学习
正在学习基础强化学习，心血来潮写了一下这个，算是学习心路历程的记录，比较简单和粗糙。效果也不是很好，有很多问题

#### 游戏环境设计
整个2048的游戏环境是我自己自己实现的，游戏的基础逻辑实现在Game.py中，直接运行Game.py即可游玩，用w、a、s、d分别代表上下左右来游玩吧！

在游戏环境的基础上实现了强化学习的环境，在Env.py中，继承了2048的Game类，加上了step和奖励的设置，我设计的奖励机制是每一步智能体合成的方块点数的log2+移动后（在移动后和游戏随机生成棋子之前）棋盘上空位置数量的$ \frac{1}{4} $，注释中还保留了很多奖励的尝试，但是感觉效果其实都差不多。另外，加上了无效移动的惩罚，无效移动会直接将这一次移动的奖励设置为-1，游戏结束时的惩罚由棋盘上最大的棋子数决定。

#### actor-critic网络设计
actor和critic网络都建立在一个基础网络BaseNet之上，BaseNet的输出经过PolicyNet和ValueNet之后分别输出一个$ batch\_size \times 4 $和$ batch\_size \times 1 $的张量，分别代表action和Q值。
BaseNet的设计如下：
首先棋盘会经过一个二进制编码，每个数字变成一个12维的向量（因为2048游戏的特殊性，二进制编码等同于one-hot编码，而由于我还没见过512以上的数字所以就先编码了12维），棋盘成为一个$ batch\_size \times 12 \times 4 \times 4 $的张量，随后经过一个两层的CNN，一层$ 3 \times 3 $和$ 1 \times 1 $的CNN，展平最后两维并交换展平后张量的最后两位，变形成$ batch\_size \times 16 \times 12 $的大小的张量，正好对应attention要求的$ batch\_size \times seq \times features $。
接下来将原始棋盘的$ batch\_size \times 12 \times 4 \times 4 $的张量也通过展平、交换后两维变形成$ batch\_size \times seq \times features $，将之前CNN的输出作为位置编码加到棋盘状态上，输入一个多头注意力，再经过多层线性层最后输出一个$ batch\_size \times num\_hidden\_output $。
之所以这么设计的理由在于CNN有提取位置信息的能力，而attention有注意整个棋盘的能力，因此采用将CNN提取的位置信息作为位置编码加到attention的输入上。

#### PPO
PPO的实现没什么好说的，就是借鉴别人的代码然后做一些修改。
但是在智能体采取行动的代码上我想做一些修改，在我PPO代码中我一开始设计的采取行动的代码是这样的：
```python
def take_action(self, state):
    try:
        state = torch.tensor([state], dtype=torch.float).to(self.device)
        probs = self.actor(state)
        action_dist = torch.distributions.Categorical(probs)
        action = action_dist.sample()
    except:
        print("NaN detected in probs, taking fallback action")
        action = torch.tensor([torch.randint(0, 4, (1,))])
    return action.item()
```
但是这样训练出来的模型在实际的测试中只会选择某两个特定的方法，经过老师的指点我顿悟，智能体在一开始中采取了某两个方向的动作后由于获得了奖励因此会增大采取这两个动作的概率，由于我categorical的建模方式，会导致采取这两个动作的概率越来越大，导致最后不会采取别的动作了，因此我在utils中实现了一个epsilon-greedy策略中的epsilon迭代器，保证模型在迭代一段时间后还能有一定概率采取随机行动，但是具体的迭代策略还没想好，因此还没投入训练，~~后续有精力了也许会更新吧~~。
```python
def take_action_with_random_choice(self, state):
    epsilon = self.epsilon_optimizer.get_epsilon()
    if random.random() < epsilon:
        action = torch.tensor([torch.randint(0, 4, (1,))])
    else:
        try:
            state = torch.tensor(
                [state], dtype=torch.float).to(self.device)
            probs = self.actor(state)
            action_dist = torch.distributions.Categorical(probs)
            action = action_dist.sample()
        except:
            print("NaN detected in probs, taking fallback action")
            action = torch.tensor([torch.randint(0, 4, (1,))])

    return action.item()
```

#### 训练参数
训练参数都在PPO_CONFIG.py中给出了。

#### 目前存在的问题
其实有一个令我很震惊的东西，我一开始训练的一个版本中看到了比较好的效果，但是后来发现网络架构除了一点问题，我在CNN输出$ batch\_size \times 12 \times 4 \times 4 $的张量后直接使用view函数变形为$ batch\_size \times 16 \times 12 $了，这样好像会导致这个16不是和棋盘每个位置的状态对应而是一种乱序的状态吧？后来我改成先将其变形为$ batch\_size \times 12 \times 16 $然后再交换后两维变成$ batch\_size \times 16 \times 12 $，这样应该才是正确的序列，但是这样子的训练效果反而变差了（可以说完全没有在训练）。
![result1](main/result.png "result1")
这是第一版我觉得网络有问题的训练过程
![result2](main/result2.png "result2")
这是第二版我将网络修改之后的训练过程，这完全就是没有在训练嘛。

有点不是太理解这是什么原因。
