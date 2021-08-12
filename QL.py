import numpy as np
import pandas as pd
import time

N_STATES = 6  # 状态
ACTIONS = ['left', 'right']  # 行为
EPSILON = 0.9  # 贪婪度
ALPHA = 0.1  # 学习度
GAMMA = 0.9  # 学习衰减度
MAX_EPISODES = 10  # 最大回合数
FRESH_TIME = 0.2  # 刷新时间

'''
    创建Q表
    行为转态
    列为行为
'''


def create_q_table(n_states, actions):
    table = pd.DataFrame(np.zeros((n_states, len(actions))), columns=actions)
    return table


'''
    在状态state下
    根据Q表和贪婪度
    选择行为
'''


def select_action(state, q_table):
    state_action = q_table.iloc[state, :]
    if (np.random.uniform() > EPSILON) or (state_action.all() == 0):
        action = np.random.choice(ACTIONS)  # 随机选择
    else:
        action = ACTIONS[state_action.argmax()]  # 贪婪选择
    return action


'''
    在状态state下
    做出action行为后
    下一个状态next_state和环境的一个反馈
    当到达终点时奖励reward=1
'''


def get_env_feedback(state, action):
    if action == 'right':  # 向右走
        if state == N_STATES - 2:
            next_state = 'terminal'  # 下一个状态为终点
            reward = 1
        else:
            next_state = state + 1
            reward = 0
    else:  # 向左走
        reward = 0
        if state == 0:
            next_state = 0
        else:
            next_state = state - 1
    return next_state, reward


'''
    更新环境
'''


def update_env(state, episode, step_counter):
    # This is how environment be updated
    env_list = ['-'] * (N_STATES - 1) + ['T']  # '---------T' our environment
    if state == 'terminal':
        interaction = 'Episode %s: total_steps = %s' % (episode + 1, step_counter)
        print('\r{}'.format(interaction), end='')
        time.sleep(2)
        print('\r                                ', end='')
    else:
        env_list[state] = 'o'
        interaction = ''.join(env_list)
        print('\r{}'.format(interaction), end='')
        time.sleep(FRESH_TIME)


'''
    Q-learning主程序
'''


def QL():
    q_table = create_q_table(N_STATES, ACTIONS)
    for episode in range(MAX_EPISODES):
        step_count = 0      # 统计每回合的步数
        state = 0
        is_terminated = False       # 回合是否结束
        update_env(state, episode, step_count)
        while not is_terminated:
            action = select_action(state, q_table)
            next_state, reward = get_env_feedback(state, action)
            q_predict = q_table.loc[state, action]      # 估计值
            if next_state != 'terminal':
                q_target = reward + GAMMA * q_table.iloc[next_state, :].max()     # 实际值（回合未结束）
            else:
                q_target = reward       # 实际值（终点）
                is_terminated = True

            q_table.loc[state, action] += ALPHA * (q_target-q_predict)        # 更新Q表
            state = next_state      # 更新状态
            step_count += 1         # 步数+1
            update_env(state, episode, step_count)      # 更新环境
    return q_table


if __name__ == '__main__':
    table = QL()
    print('\r\nQ-table:\n')
    print(table)
