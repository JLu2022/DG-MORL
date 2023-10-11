import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

expected_utility_list = np.load("expected_utility_list.npy", allow_pickle=True)
expected_utility_list_gpi = np.load("mean_utility_gpi-ls.npy", allow_pickle=True)
expected_utility_list_gpi_pd = np.load("mean_utility_gpi_pd.npy", allow_pickle=True)
# print(expected_utility_list_gpi)
# print(expected_utility_list)
print(len(expected_utility_list))
# for li in expected_utility_list:
#     print(f"len:{len(li)}\t li:{li}")
print(f"JSMORL-mean_u:{np.mean(expected_utility_list, axis=0)[:40000]}")
print(len(np.mean(expected_utility_list, axis=0)))

print(f"GPI-mean_u:{expected_utility_list_gpi}")
print(len(expected_utility_list_gpi))
# plt.plot(np.mean(expected_utility_list, axis=0))
x = np.mean(expected_utility_list, axis=0)  # 横轴数据，1k到10k
x1 = expected_utility_list_gpi
x2 = expected_utility_list_gpi_pd
# y = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])  # 示例纵轴数据

# 创建一个图形和坐标轴
fig, ax = plt.subplots()


def format_func(value, tick_number):
    return f'{int(value / 4000)}'
ax.xaxis.set_major_formatter(FuncFormatter(format_func))
if len(x) < len(x1):
    limit = len(x)
else:
    limit = 40000
# limit = 10000
# 绘制线形图
ax.plot(x[:limit], label="JSMORL(ours)")
ax.plot(x1[:limit], color="red",label="GPI-LS")
ax.plot(x2[:limit], color="green",label="GPI-PD")
# 添加标题和标签
ax.set_title('Expected Utility Comparison (DST Environment)')
ax.set_xlabel('Time Step (4*10^3)')
ax.set_ylabel('Expected Utility')
# plt.ylim(3, 5.63)
# 显示图形
plt.legend()
plt.show()
