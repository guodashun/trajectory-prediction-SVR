import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d


def plt_show(data, num=1, color=['red']):
    ax = plt.subplot(projection='3d')  # 创建一个三维的绘图工程
    ax.set_title('3d_image_show')  # 设置本图名称
    ax.set_xlabel('X')  # 设置x坐标轴
    ax.set_ylabel('Y')  # 设置y坐标轴
    ax.set_zlabel('Z')  # 设置z坐标轴
    for i in range(num):
        ax.scatter(data[i][0], data[i][1], data[i][2], color=color[i], s=1)
        # ax.plot(data[i][0], data[i][1], data[i][2], color=color[i])
    plt.show()
