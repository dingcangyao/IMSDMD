from cyvlfeat.kmeans import kmeans, kmeans_quantize
import numpy as np
from multiprocessing.pool import Pool


from computeSDMD import computeSDMD


def trainVLADCodeBook(images, dmd_options, kmeans_options):
    max_descriptor = kmeans_options['num_descriptor']
    num_cluster = kmeans_options['num_cluster']
    num_images = len(images)
    num_descriptor_per_image = int(np.ceil(max_descriptor / num_images))

    descrs = []
    pool = Pool(processes=8)
    # features = computeDMD(img, dmd_options)
    mul_features = [pool.apply_async(computeSDMD, args=(img, dmd_options, 0)) for img in images]
    pool.close()#先获取测试图像的SDMD特征，然后进行聚类
    pool.join()
    print('进入HVLAD')
    for features in mul_features:
        # print("[trainVLADCodeBook]reading:path='{}' features_shape:{}".format(img, features.shape))
        features = features.get()
        sel = list(np.random.permutation(features.shape[1]))[: num_descriptor_per_image]
        descrs = features[:, sel] if len(descrs) == 0 else np.concatenate((descrs, features[:, sel]), axis=1)

    new_descrs = np.zeros((max_descriptor, descrs.shape[0]), dtype=np.float32)
    new_descrs[:, :] = descrs.T[:max_descriptor, :]

    centers = kmeans(new_descrs, num_centers=num_cluster, verbose=True, initialization='PLUSPLUS',
                     min_energy_variation=0.000001).astype(np.float32)


    return {'centers': centers}
