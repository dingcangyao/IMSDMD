import numpy as np


def computerCoordinates1(block_radii, n_points, n_folds):
    """
    Parameters
    ----------
    block_radii: 微块半径
    n_points: 微块对数量

    Returns
    -------
    xi: 第一个微块坐标
    yi: 相对应的另一个微块坐标
    """
    xi = []
    yi = []
    block_size = 2 * block_radii + 1#半径是7，块尺寸是15
    point_per_fold = int(n_points / n_folds)#每一个旋转度数的点就是10个
    test1=np.sqrt((block_size**2)/25)
    test3=(2,point_per_fold)
    test2=np.random.normal(0,test1,test3)
    test4=np.round(test2);
    pts1 = np.round(np.random.normal(0, np.sqrt((block_size ** 2) / 25), (2, 80)))#利用方差是3.0的标准正态分布来生成（2，10）矩阵，就是
    # （x，y）坐标10个，这个生成的是Xi
    # 下面是对坐标进行最大值最小值处理
    pts1[pts1 > block_radii] = block_radii
    pts1[pts1 < -block_radii] = -block_radii
    xi = pts1 if len(xi) == 0 else np.concatenate((xi, pts1), axis=1)

    pts2 = np.round(np.random.normal(0, np.sqrt((block_size ** 2) / 25), (2, 80)))#这个生成的是Yi
    pts2[pts2 > block_radii] = block_radii
    pts2[pts2 < -block_radii] = -block_radii
    yi = pts2 if len(yi) == 0 else np.concatenate((yi, pts2), axis=1)
    #去掉重复的样本点，相同的就去掉
    t1=np.sum((xi == yi).astype(int), axis=0)
    indices = np.where(np.sum((xi == yi).astype(int), axis=0) == 2)
    np.delete(xi, indices, axis=1)
    np.delete(yi, indices, axis=1)


    return xi, yi
# block_radius=7
# nPoints=80
# nFolds=8
# computerCoordinates(7,80,8)