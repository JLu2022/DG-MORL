import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
expected_utility_list = np.load("expected_utility_list.npy", allow_pickle=True)
print(expected_utility_list)
print(np.mean(expected_utility_list, axis=0))

plt.plot(np.mean(expected_utility_list, axis=0))
x = np.mean(expected_utility_list, axis=0)   # 横轴数据，1k到10k
# y = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])  # 示例纵轴数据

# 创建一个图形和坐标轴
fig, ax = plt.subplots()

# 自定义函数来格式化刻度标签
def format_func(value, tick_number):
    # 将数值除以1000并附加'k'来表示千
    return f'{int(value/4000)}'

# 应用自定义格式化函数到横轴
ax.xaxis.set_major_formatter(FuncFormatter(format_func))

# 绘制线形图
ax.plot(x)

# 添加标题和标签
ax.set_title('Line Plot with Custom X-axis')
ax.set_xlabel('X-axis (in 4*10^3)')

# 显示图形
plt.show()
plt.show()

