import matplotlib.pyplot as plt
import numpy as np
def PlotDemo1(xi,yi,type):
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    le=xi.shape[1]
    block_radius=7
    max_scale=4
    if type==0:
        xi_c=xi
        yi_c=yi
        xi_c[xi_c > block_radius - max_scale + 1] = block_radius - max_scale + 1
        xi_c[xi_c < -block_radius] = -block_radius
        yi_c[yi_c > block_radius - max_scale + 1] = block_radius - max_scale + 1
        yi_c[yi_c < -block_radius] = -block_radius
        # 把坐标都挪到图片里面
        xi_c = (xi_c + block_radius + 1).astype(np.uint16)
        yi_c = (yi_c + block_radius + 1).astype(np.uint16)
        for i in range(le):


            ax.scatter(xi_c[0,i], xi_c[1,i], s=10, c='blue', marker='.')
            ax.scatter(yi_c[0,i], yi_c[1,i], s=10, c='orange', marker='.')
            ax.plot([xi_c[0,i], yi_c[0,i]], [xi_c[1,i], yi_c[1,i]], c='violet')
    else:
        for i in range(le):


            ax.scatter(xi[0,i], xi[1,i], s=10, c='blue', marker='.')
            ax.scatter(yi[0,i], yi[1,i], s=10, c='orange', marker='.')
            ax.plot([xi[0,i], yi[0,i]], [xi[1,i], yi[1,i]], c='violet')

    plt.show()

