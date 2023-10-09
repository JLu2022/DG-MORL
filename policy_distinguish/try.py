import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

expected_utility_list = np.load("expected_utility_list.npy", allow_pickle=True)
expected_utility_list_gpi = np.load("mean_utility.npy", allow_pickle=True)
# print(expected_utility_list_gpi)
# print(expected_utility_list)
print(f"JSMORL-mean_u:{np.mean(expected_utility_list, axis=0)[:40000]}")
print(len(np.mean(expected_utility_list, axis=0)))

print(f"GPI-mean_u:{expected_utility_list_gpi}")
print(len(expected_utility_list_gpi))
# plt.plot(np.mean(expected_utility_list, axis=0))
x = np.mean(expected_utility_list, axis=0)  # 横轴数据，1k到10k
x1 = expected_utility_list_gpi
# y = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])  # 示例纵轴数据

# 创建一个图形和坐标轴
fig, ax = plt.subplots()


# 自定义函数来格式化刻度标签
def format_func(value, tick_number):
    # 将数值除以1000并附加'k'来表示千
    return f'{int(value / 4000)}'


# 应用自定义格式化函数到横轴
ax.xaxis.set_major_formatter(FuncFormatter(format_func))

# 绘制线形图
ax.plot(x[:40000])
ax.plot(x1[:40000],color="red")
# 添加标题和标签
ax.set_title('Line Plot with Custom X-axis')
ax.set_xlabel('X-axis (in 4*10^3)')
# plt.ylim(3, 5.63)
# 显示图形
plt.show()
# # plt.show()
# 定义移动平均的窗口大小
# window_size = 1000
#
# # 计算移动平均
# moving_average = np.convolve(np.mean(expected_utility_list, axis=0), np.ones(window_size)/window_size, mode='valid')
# moving_average_gpi = np.convolve(expected_utility_list_gpi, np.ones(window_size)/window_size, mode='valid')
#
# # 创建一个图形和坐标轴
# fig, ax = plt.subplots()
#
# # 自定义函数来格式化刻度标签
# def format_func(value, tick_number):
#     return f'{int(value / 4000)}'
#
# # 应用自定义格式化函数到横轴
# ax.xaxis.set_major_formatter(FuncFormatter(format_func))
#
# # 绘制移动平均线形图
# ax.plot(moving_average[:40000], label='Moving Average (Your Data)')
# ax.plot(moving_average_gpi[:40000], color="red", label='Moving Average (GPI Data)')
#
# # 添加标题和标签
# ax.set_title('Moving Average Line Plot')
# ax.set_xlabel('X-axis (in 4*10^3)')
#
# # 添加图例
# plt.legend()
# plt.ylim(0, 7)
# # 显示图形
# plt.show()