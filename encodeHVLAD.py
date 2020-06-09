from multiprocessing.pool import Pool

import numpy as np
from cyvlfeat.kmeans import kmeans_quantize
from computeSDMD import computeSDMD


def encodeHVLAD(images, encoder, dmd_options):
    l_descrs, m_descrs, s_descrs, all_descrs = [], [], [], []

    pool = Pool(processes=4)
    l_features = [pool.apply_async(computeSDMD, args=(img, dmd_options, 0)) for img in images]#零层高斯金字塔出现的图片特征
    m_features = [pool.apply_async(computeSDMD, args=(img, dmd_options, 1)) for img in images]#1层高斯金字塔出现的图片特征
    s_features = [pool.apply_async(computeSDMD, args=(img, dmd_options, 2)) for img in images]#2层高斯金字塔出现的图片特征
    pool.close()
    pool.join()

    centers = encoder['centers']
    # vars = encoder['vars']
    # skews = encoder['skews']
#以下的循环，第一步：循环每一次的变量，就是从上面通过computeSDMD获得的金字塔一层中，一个图片的SDMD特征（８０，８６４９），把它转置命名为ｆｅａｔｕｒｅｓ，和ｎｅｗｆｅａｔｕｒｅｓ
    # 第二步：为每个特征和聚类中心索引上，就有了predicted_labels（１，８６４９）
    # 把每个类中心索引的特征值单拿出来，计算Ｖｍ，特征值的平均值和聚类中心相减再乘上索引到当前聚类中心的特征个数，计算Ｖｃ，
    # Ｖｃ就是当前聚类中心特征的方差，减去以聚类中心作为均值的方差，最后按坐标累加再除以特征个数，每个聚类中心都会形成（１，８０），然后经过
    # １２８循环，形成（１２８，８０）
    # Ｖ＿ａｌｌ就是把Ｖｍ和Ｖｃ拼接在一起，然后再转化成一维（１，１２８＊８０）
    # 最后经过４０次循环，形成（４０，１２８＊８０）这个就是一层高斯金字塔的ｅｎｃｏｄｅ
    # 接下来的另外两个循环也是一样的，只不过就是再高斯金字塔的更高层而已，计算完ｌ，ｓ，ｍ也就是１.２.３层的ｖｌａｄ编码，再把这三个放在一起取平均值
    # 得到的ｄｅｃｒｉｓ
    print('进入HVLAD')
    for features in l_features:
        features = features.get().T#图片特征就是（80，8６49），经过转置，图片特征是（8６49，80）

        new_features = np.zeros((features.shape[0], features.shape[1]), dtype=np.float32)#全零矩阵(8649,80)
        new_features[:, :] = features[:, :]#把features的值赋给new_features

        predicted_labels = kmeans_quantize(data=new_features, centers=centers)#就是聚类为每个聚类中心分配特征，或者说为每一行的特征找索引
        n_cluster = centers.shape[0]#center（128，80）
        [n_patch, n_feature] = features.shape
#以上是对图片的特征进行与聚类中心的索引
        Vm = np.zeros([n_cluster, n_feature], dtype=np.float32)#Vm（128，80）
        Vc = np.zeros([n_cluster, n_feature], dtype=np.float32)#vc （128，80）
        # Vs = np.zeros([n_cluster, n_feature], dtype=np.float32)
        for i in range(n_cluster):
            Ni = np.sum(predicted_labels == i)
            if Ni > 0:
                i_features = features[predicted_labels == i, :] #挑选相应的列，（  Ni，80）
                mi = np.mean(i_features, axis=0)#mi （1，80）
                Vm[i] = Ni * (mi - centers[i])#特征与聚类中心相减然后再乘上使用这个聚类中心的索引到聚类中心的个数
                Vc[i] = (1 / Ni) * np.sum((i_features - mi) ** 2, axis=0) - (1 / Ni) * np.sum(
                    (i_features - centers[i]) ** 2, axis=0)#前面的np.sum是先计算ifeatures每一行的值减去均值，然后平方累加，好像就是求方差，第二个就是把聚类中心当作均值来求方差
                #上面应该是不同均值计算的方差均值相减
                # Vs[i] = ((1 / Ni) * (np.sum((i_features - mi) ** 3, axis=0))) / np.maximum(
                #     ((1 / Ni) * np.sum((i_features - mi) ** 2, axis=0)) ** 1.5, 1e-12) - (
                #                 (1 / Ni) * (np.sum((i_features - centers[i]) ** 3, axis=0))) / np.maximum(
                #     ((1 / Ni) * np.sum((i_features - centers[i]) ** 2, axis=0)) ** 1.5, 1e-12)
                #
        # power normalization, also called square-rooting normalization
        Vm = np.sign(Vm) * np.sqrt(np.abs(Vm))
        Vc = np.sign(Vc) * np.sqrt(np.abs(Vc))
        # Vs = np.sign(Vs) * np.sqrt(np.abs(Vs))
        # # L2 normalization
        # Vm /= np.maximum(np.linalg.norm(Vm, axis=1)[:, None], 1e-12)
        # Vc /= np.maximum(np.linalg.norm(Vc, axis=1)[:, None], 1e-12)
        # Vs /= np.maximum(np.linalg.norm(Vs, axis=1)[:, None], 1e-12)
        # V_all = np.vstack((Vm, Vc, Vs)).flatten()[None, :]
        V_all = np.vstack((Vm, Vc)).flatten()[None, :]#拼接到一起，先是合并到一起（128，160），然后转成一维
        l_descrs = V_all if len(l_descrs) == 0 else np.concatenate((l_descrs, V_all), axis=0)

    for features in m_features:
        features = features.get().T

        new_features = np.zeros((features.shape[0], features.shape[1]), dtype=np.float32)
        new_features[:, :] = features[:, :]
        predicted_labels = kmeans_quantize(data=new_features, centers=centers)
        n_cluster = centers.shape[0]
        [n_patch, n_feature] = features.shape

        Vm = np.zeros([n_cluster, n_feature], dtype=np.float32)
        Vc = np.zeros([n_cluster, n_feature], dtype=np.float32)
        # Vs = np.zeros([n_cluster, n_feature], dtype=np.float32)
        for i in range(n_cluster):
            Ni = np.sum(predicted_labels == i)
            if Ni > 0:
                i_features = features[predicted_labels == i, :]
                mi = np.mean(i_features, axis=0)
                Vm[i] = Ni * (mi - centers[i])
                Vc[i] = (1 / Ni) * np.sum((i_features - mi) ** 2, axis=0) - (1 / Ni) * np.sum(
                    (i_features - centers[i]) ** 2, axis=0)
                # Vs[i] = ((1 / Ni) * (np.sum((i_features - mi) ** 3, axis=0))) / np.maximum(
                #     ((1 / Ni) * np.sum((i_features - mi) ** 2, axis=0)) ** 1.5, 1e-12) - (
                #                 (1 / Ni) * (np.sum((i_features - centers[i]) ** 3, axis=0))) / np.maximum(
                #     ((1 / Ni) * np.sum((i_features - centers[i]) ** 2, axis=0)) ** 1.5, 1e-12)
        # power normalization, also called square-rooting normalization
        Vm = np.sign(Vm) * np.sqrt(np.abs(Vm))
        Vc = np.sign(Vc) * np.sqrt(np.abs(Vc))
        # Vs = np.sign(Vs) * np.sqrt(np.abs(Vs))
        # # L2 normalization
        # Vm /= np.maximum(np.linalg.norm(Vm, axis=1)[:, None], 1e-12)
        # Vc /= np.maximum(np.linalg.norm(Vc, axis=1)[:, None], 1e-12)
        # Vs /= np.maximum(np.linalg.norm(Vs, axis=1)[:, None], 1e-12)
        # V_all = np.vstack((Vm, Vc, Vs)).flatten()[None, :]
        V_all = np.vstack((Vm, Vc)).flatten()[None, :]
        m_descrs = V_all if len(m_descrs) == 0 else np.concatenate((m_descrs, V_all), axis=0)

    for features in s_features:
        features = features.get().T

        new_features = np.zeros((features.shape[0], features.shape[1]), dtype=np.float32)
        new_features[:, :] = features[:, :]
        predicted_labels = kmeans_quantize(data=new_features, centers=centers)
        n_cluster = centers.shape[0]
        [n_patch, n_feature] = features.shape

        Vm = np.zeros([n_cluster, n_feature], dtype=np.float32)
        Vc = np.zeros([n_cluster, n_feature], dtype=np.float32)
        # Vs = np.zeros([n_cluster, n_feature], dtype=np.float32)
        for i in range(n_cluster):
            Ni = np.sum(predicted_labels == i)
            if Ni > 0:
                i_features = features[predicted_labels == i, :]
                mi = np.mean(i_features, axis=0)
                Vm[i] = Ni * (mi - centers[i])
                Vc[i] = (1 / Ni) * np.sum((i_features - mi) ** 2, axis=0) - (1 / Ni) * np.sum(
                    (i_features - centers[i]) ** 2, axis=0)
                # Vs[i] = ((1 / Ni) * (np.sum((i_features - mi) ** 3, axis=0))) / np.maximum(
                #     ((1 / Ni) * np.sum((i_features - mi) ** 2, axis=0)) ** 1.5, 1e-12) - (
                #                 (1 / Ni) * (np.sum((i_features - centers[i]) ** 3, axis=0))) / np.maximum(
                #     ((1 / Ni) * np.sum((i_features - centers[i]) ** 2, axis=0)) ** 1.5, 1e-12)
        # power normalization, also called square-rooting normalization
        Vm = np.sign(Vm) * np.sqrt(np.abs(Vm))
        Vc = np.sign(Vc) * np.sqrt(np.abs(Vc))
        # Vs = np.sign(Vs) * np.sqrt(np.abs(Vs))
        # # L2 normalization
        # Vm /= np.maximum(np.linalg.norm(Vm, axis=1)[:, None], 1e-12)
        # Vc /= np.maximum(np.linalg.norm(Vc, axis=1)[:, None], 1e-12)
        # Vs /= np.maximum(np.linalg.norm(Vs, axis=1)[:, None], 1e-12)
        # V_all = np.vstack((Vm, Vc, Vs)).flatten()[None, :]
        V_all = np.vstack((Vm, Vc)).flatten()[None, :]
        s_descrs = V_all if len(s_descrs) == 0 else np.concatenate((s_descrs, V_all), axis=0)

    descrs = (l_descrs + m_descrs + s_descrs)/3
    return descrs.astype(np.float32)
