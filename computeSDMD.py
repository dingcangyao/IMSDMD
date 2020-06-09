import numpy as np
import cv2

from scipy import signal

from computeCoordinates import computerCoordinates
from dimensionalityReduction import dimensionalityReduction

def computeSDMD(img_path, dmd_options, level):
    """
    Parameters
    ----------
    img_path: 图像路径 dtypes = list
    dmd_options: DMD特征参数 dtypes = dict

    Returns
    -------
    v.T: 一行是一个微块对数据 n_points * X X是图像patch数量
    """
    pts1 = dmd_options['xi']
    pts2 = dmd_options['yi']
    n_folds = int(dmd_options['n_folds'])
    block_radius = int(dmd_options['block_radius'])
    grid_space = int(dmd_options['grid_space'])
    max_scale = int(dmd_options['scale'])
    n_components = int(dmd_options['n_components'])

    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)#Kth_tips图片是（200，200）
    img=cv2.resize(img,(200,200))
    if level != 0:#就是利用高斯模糊，减尺度
        for i in range(level):
            img = cv2.pyrDown(img)#进行高斯金字塔构建，把分辨率还有图片尺寸都降一个等级
    # x = cv2.Sobel(img, cv2.CV_16S, 1, 0)
    # y = cv2.Sobel(img, cv2.CV_16S, 0, 1)
    #
    # absX = cv2.convertScaleAbs(x)  # 转回uint8
    # absY = cv2.convertScaleAbs(y)
    #
    # dst = cv2.addWeighted(absX, 0.5, absY, 0.5, 0) 梯度值图片
    # pts1[pts1 > block_radius - max_scale + 1] = block_radius - max_scale + 1
    # pts1[pts1 < -block_radius] = -block_radius
    # pts2[pts2 > block_radius - max_scale + 1] = block_radius - max_scale + 1
    # pts2[pts2 < -block_radius] = -block_radius
    #
    # # 把坐标都挪到图片里面
    # pts1 = (pts1 + block_radius + 1).astype(np.uint16)
    # pts2 = (pts2 + block_radius + 1).astype(np.uint16)
    # 用来筛选图片目标
    # for i in range(8):
    #     a1=pts1[ :,i*10:(i+1)*10]
    #     a2=pts2[ :,i*10:(i+1)*10]
    #     s1 = np.zeros(10)
    #     s2 = np.zeros(10)
    #     for j in range(10):
    #         sob_1=dst[a1[0,j],a1[1,j]]
    #         sob_2=dst[a2[0,j],a2[1,j]]
    #         s1[j]=sob_1
    #         s2[j]=sob_2
    #     x_max=np.argmax(s1)
    #     x_min=np.argmin(s1)
    #     y_max=np.argmax(s2)
    #     y_min=np.argmin(s2)
    #     pts2[:,[i*10+x_max,i*10+y_max]]=pts2[:,[i*10+y_max,i*10+x_max]]
    #     pts2[:,[i*10+x_min,i*10+y_min]]=pts2[:,[i*10+y_min,i*10+x_min]]

    d_img = img.astype(np.float32) / 255#转换成了0~1的数组

    num_sample = pts1.shape[1]#采样点，也就是坐标有80个
    sample_per_scale = num_sample / max_scale#80个采样
    #下面对坐标进行处理，把最大值和最小值都控制好，因为之前的处理仅仅是对前十个原始坐标进行了处理，没有对后续的进行处理
    # pts1[pts1 > block_radius - max_scale + 1] = block_radius - max_scale + 1
    # pts1[pts1 < -block_radius] = -block_radius
    # pts2[pts2 > block_radius - max_scale + 1] = block_radius - max_scale + 1
    # pts2[pts2 < -block_radius] = -block_radius
    #
    # #把坐标都挪到图片里面
    # pts1 = (pts1 + block_radius + 1).astype(np.uint16)
    # pts2 = (pts2 + block_radius + 1).astype(np.uint16)

    block_size = 2 * block_radius + 1#块的尺寸是15，应该就是论文中写的patch
    nPoints = pts1.shape[1]#一共有80个点
    img_shape = img.shape#记录图片的尺寸（200，200）
    effect_row = img_shape[0] - block_size#有效的行，就是去掉微块的大小，因为坐标取值是在这里面取的值，185
    effect_col = img_shape[1] - block_size#有效的列，就是去掉微块的大小，因为坐标取值是在这里面取的值，185
    v = []#用来存放sdmd特征

    itimg = np.cumsum(d_img, axis=0, dtype=np.float32)#对图片像素值进行行累加，就是每一行的值，都等于这一行加上上一行的值
    itimg = np.cumsum(itimg, axis=1, dtype=np.float32)#对矩阵进行列累加，操作就和上面的行累加一样

    # iimg = np.zeros(np.array(img_shape) + 2,dtype=np.uint32)
    # iimg[1:iimg.shape[0] - 1,1:iimg.shape[1] - 1] = itimg
    iimg = cv2.copyMakeBorder(itimg, 1, 1, 1, 1, cv2.BORDER_CONSTANT, 0)#就是给前面的矩阵加上一层边框，（202，202）
    # norm_matirx = np.sqrt(cv2.filter2D(d_img ** 2, -1, np.ones((block_size, block_size), dtype=np.double)))
    norm_matirx = np.sqrt(signal.correlate2d(d_img ** 2, np.ones((block_size, block_size), dtype=np.float32), mode='same'))
    # 先把d_img(就是那个图像转换成的浮点数）每个值平方，然后15*15的单位矩阵进行滤波，其实就是求和（200，200）的矩阵
    norm_matirx = norm_matirx[block_radius: norm_matirx.shape[0] - block_radius,
                  block_radius: norm_matirx.shape[1] - block_radius]
    #对norm_matirx进行截取，就是把行列两头都去掉7个（186，186）

    for i in range(nPoints):
        # 上面是计算 Xi对应的图像特征值，下面是计算Yi的
        micro_block_size = np.int(np.floor((i + sample_per_scale) / sample_per_scale))#微块的大小
        iiPt1 = iimg[pts1[0, i] + micro_block_size - 1: pts1[0, i] + effect_row + micro_block_size,
                pts1[1, i] + micro_block_size - 1: pts1[1, i] + effect_col + micro_block_size]
        #截取（186，186）个像素，从点开始往左往下延申185个像素
        iiPt2 = iimg[pts1[0, i] + micro_block_size - 1: pts1[0, i] + effect_row + micro_block_size,
                pts1[1, i] - 1: pts1[1, i] + effect_col]
        # 截取（186，186）个像素，从点的左边microblock个点开始往左往下延申185个像素
        iiPt3 = iimg[pts1[0, i] - 1: pts1[0, i] + effect_row,
                pts1[1, i] + micro_block_size - 1: pts1[1, i] + effect_col + micro_block_size]
        # 截取（186，186）个像素，从上面microblock个点开始往左往下延申185个像素
        iiPt4 = iimg[pts1[0, i] - 1: pts1[0, i] + effect_row, pts1[1, i] - 1: pts1[1, i] + effect_col]
        ## 截取（186，186）个像素，从左上面microblock个点开始往左往下延申185个像素
        block_sum_x = iiPt4 + iiPt1 - iiPt2 - iiPt3#就是把这四个矩阵里的值相加再相减

        iiPt1 = iimg[pts2[0, i] + micro_block_size - 1: pts2[0, i] + effect_row + micro_block_size,
                pts2[1, i] + micro_block_size - 1: pts2[1, i] + effect_col + micro_block_size]
        iiPt2 = iimg[pts2[0, i] + micro_block_size - 1: pts2[0, i] + effect_row + micro_block_size,
                pts2[1, i] - 1: pts2[1, i] + effect_col]
        iiPt3 = iimg[pts2[0, i] - 1: pts2[0, i] + effect_row,
                pts2[1, i] + micro_block_size - 1: pts2[1, i] + effect_col + micro_block_size]
        iiPt4 = iimg[pts2[0, i] - 1: pts2[0, i] + effect_row, pts2[1, i] - 1: pts2[1, i] + effect_col]

        block_sum_y = iiPt4 + iiPt1 - iiPt2 - iiPt3

        block_sum_x = block_sum_x / (micro_block_size ** 2)#把得来的点的微块值，再除以微块大小的平方
        block_sum_y = block_sum_y / (micro_block_size ** 2)

        diff_micro_block = (block_sum_x - block_sum_y) / norm_matirx#把Xi-Yi，再除以这个矩阵（186，186）

        selected_grid = (diff_micro_block[0:diff_micro_block.shape[0]:grid_space,
                         0:diff_micro_block.shape[1]:grid_space]).flatten()[:, None]#先对微块差隔一行取一行，隔一列取一列，然后再进行一维化
        # （1，8649）

        v = selected_grid if len(v) == 0 else np.concatenate((v, selected_grid), axis=1)

    max_sum_diff = []
    # 接下来就是对行列进行调整，把最大值的一组folds放到最卡面
    for i in range(n_folds):
        start = int(i * nPoints / n_folds)
        end = int((i + 1) * nPoints / n_folds)
        sum_diff_per_fold = np.sum(v[:, start:end], axis=1)[:, None]#列相加(8649,1)
        max_sum_diff = sum_diff_per_fold if len(max_sum_diff) == 0 else np.concatenate(
            (max_sum_diff, sum_diff_per_fold), axis=1)
    idxs = np.argmax(max_sum_diff, axis=1)

    for row, index in enumerate(idxs):
        if index != 0:
            start = int(index * (nPoints / n_folds))
            [end] = v[row, :].shape
            v[row, :] = np.concatenate((v[row, start:end], v[row, 0:start]))

    # v = dimensionalityReduction(v.T, n_components)
    print("[MSDMD]path='{}' shape={}".format(img_path, v.shape))

    return v.T
# dmd_options, kmeans_options = {}, {}
# block_radius = 7
# nPoints = 80
# n_fold = 8
# xi, yi = computerCoordinates(block_radius, nPoints, n_fold)
# grid_space = 2
# scale = 4
# crossValIndex = 10
# dmd_options['block_radius'] = block_radius
# dmd_options['nPoints'] = nPoints
# dmd_options['xi'] = xi
# dmd_options['yi'] = yi
# dmd_options['n_folds'] = n_fold
# dmd_options['grid_space'] = grid_space
# dmd_options['scale'] = scale
# dmd_options['n_components'] = 40
# dmd_options['levels'] = 2
# computeSDMD('data/KTH_TIPS/aluminium_foil/15-scale_1_im_1_grey.png',dmd_options,0)