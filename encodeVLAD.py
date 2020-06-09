from multiprocessing.pool import Pool

import numpy as np
from cyvlfeat.kmeans import kmeans_quantize
from computeSDMD import computeSDMD


def encodeVLAD(images, encoder, dmd_options):
    descrs = []

    pool = Pool(processes=4)
    features = [pool.apply_async(computeSDMD, args=(img, dmd_options, 0)) for img in images]
    pool.close()
    pool.join()

    centers = encoder['centers']

    for feature in features:
        feature = feature.get().T

        new_feature = np.zeros((feature.shape[0], feature.shape[1]), dtype=np.float32)
        new_feature[:, :] = feature[:, :]
        predicted_labels = kmeans_quantize(data=new_feature, centers=centers)
        n_cluster = centers.shape[0]
        [n_patch, n_feature] = new_feature.shape

        Vm = np.zeros([n_cluster, n_feature], dtype=np.float32)
        Vc = np.zeros([n_cluster, n_feature], dtype=np.float32)
        Vs = np.zeros([n_cluster, n_feature], dtype=np.float32)
        for i in range(n_cluster):
            Ni = np.sum(predicted_labels == i)
            if Ni > 0:
                i_features = new_feature[predicted_labels == i, :]
                mi = np.mean(i_features, axis=0)
                Vm[i] = Ni * (mi - centers[i])
                Vc[i] = (1 / Ni) * np.sum((i_features - mi) ** 2, axis=0) - (1 / Ni) * np.sum(
                    (i_features - centers[i]) ** 2, axis=0)
                Vs[i] = ((1 / Ni) * (np.sum((i_features - mi) ** 3, axis=0))) / np.maximum(
                    ((1 / Ni) * np.sum((i_features - mi) ** 2, axis=0)) ** 1.5, 1e-12) - (
                                (1 / Ni) * (np.sum((i_features - centers[i]) ** 3, axis=0))) / np.maximum(
                    ((1 / Ni) * np.sum((i_features - centers[i]) ** 2, axis=0)) ** 1.5, 1e-12)
        # power normalization, also called square-rooting normalization
        Vm = np.sign(Vm) * np.sqrt(np.abs(Vm))
        Vc = np.sign(Vc) * np.sqrt(np.abs(Vc))
        Vs = np.sign(Vs) * np.sqrt(np.abs(Vs))
        # # L2 normalization
        # Vm /= np.maximum(np.linalg.norm(Vm, axis=1)[:, None], 1e-12)
        # Vc /= np.maximum(np.linalg.norm(Vc, axis=1)[:, None], 1e-12)
        # Vs /= np.maximum(np.linalg.norm(Vs, axis=1)[:, None], 1e-12)

        V_all = np.vstack((Vm, Vc, Vs)).flatten()[None, :]

        descrs = V_all if len(descrs) == 0 else np.concatenate((descrs, V_all), axis=0)
    return descrs.astype(np.float32)
1