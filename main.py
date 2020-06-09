import numpy as np
import os
import shutil
from draw import PlotDemo1
from coordianteFelt import coordianteFelt
from computeCoordinates import computerCoordinates
from encodeHVLAD import encodeHVLAD
from encodeHVLAD_ import encodeHVLAD_
from encodeVLAD import encodeVLAD
from load_img import get_images
from cc import computerCoordinates1
from trainVLADCodeBook import trainVLADCodeBook
from sklearn import svm
from sklearn import model_selection
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

if __name__ == '__main__':

    dmd_options, kmeans_options = {}, {}
    block_radius = 7
    nPoints = 80
    n_fold = 8
    xi, yi = computerCoordinates(block_radius, nPoints, n_fold)
    # x2,y2=computerCoordinates1(block_radius, nPoints, n_fold)
    # PlotDemo1(x2,y2,0)
    # PlotDemo1(xi,yi,0)
    grid_space = 2
    scale = 4
    crossValIndex = 10
    dmd_options['block_radius'] = block_radius
    dmd_options['nPoints'] = nPoints
    dmd_options['xi'] = xi
    dmd_options['yi'] = yi
    dmd_options['n_folds'] = n_fold
    dmd_options['grid_space'] = grid_space
    dmd_options['scale'] = scale
    dmd_options['n_components'] = 40
    dmd_options['levels'] = 2

    kmeans_options['num_descriptor'] = 500000
    kmeans_options['num_cluster'] = 128

    datasets = ['curet','umd','KTH_TIPS']
    n_train = {'KTH_TIPS': 40 * 10, 'curet': 102* 9, 'umd': 20 * 10, 'uiuc': 20 * 25}
    n_test = {'KTH_TIPS': 41 * 10, 'curet': 103 * 9, 'umd': 20* 10, 'uiuc': 20 * 25}
    for dataset in datasets:

        experiment_path = os.path.join('./experiments', dataset)
        if os.path.exists(experiment_path) and len(os.listdir(experiment_path)) == crossValIndex:#之所以要等于10，就是因为要对一个图片集进行10次循环
            acc = []
            print("%s dataset:" % (dataset))
            for i in range(crossValIndex):
                score = np.load('./experiments/{}/dmd-seed-{}/score.npy'.format(dataset, i))
                print("dmd-seed-%s: %.2f%%" % (i, score * 100))
                acc.append(score)
            print("mean acc %.2f%%" % (np.mean(acc) * 100))
            print("mean std %.2f%%" % (np.std(acc) * 100))
            continue
        else:

            if os.path.exists(experiment_path):
                shutil.rmtree(experiment_path)
            print("{} dataset training:".format(dataset))

            images_list, labels_list = get_images(dataset)

            shufflesplit = model_selection.StratifiedShuffleSplit(n_splits=crossValIndex, train_size=n_train[dataset],
                                                                  test_size=n_test[dataset])

            i = 0
            for train_index, test_index in shufflesplit.split(images_list, labels_list):
                # 会循环10次，n_splits就是分成这些数目，crossValIndex=10，所以就会分10次
                X_train = images_list[train_index]
                y_train = labels_list[train_index]
                X_test = images_list[test_index]
                y_test = labels_list[test_index]
                print("X_train:{} y_train:{} X_test:{} y_test:{}".format(X_train.shape, y_train.shape, X_test.shape,
                                                                         y_test.shape))
                x1,y1=coordianteFelt(X_train,xi,yi)
                # PlotDemo1(x1, y1,1)
                dmd_options['xi'] = x1
                dmd_options['yi'] = y1
                encoder = trainVLADCodeBook(X_train, dmd_options, kmeans_options)

                train_descrs = encodeHVLAD(images=X_train, dmd_options=dmd_options, encoder=encoder)
                # 三层金字塔的描述符合并在一起了
                test_descrs = encodeHVLAD(images=X_test, dmd_options=dmd_options, encoder=encoder)
                # 测试的VLAD描述符
                # test_descrs = encodeVLAD(X_test, encoder, dmd_options)
                # train_descrs = encodeVLAD(X_train, encoder, dmd_options)

                os.makedirs(os.path.join(experiment_path, 'dmd-seed-{}'.format(i)))

                np.save('./experiments/{}/dmd-seed-{}/xi.npy'.format(dataset, i), xi)
                np.save('./experiments/{}/dmd-seed-{}/yi.npy'.format(dataset, i), yi)
                np.save('./experiments/{}/dmd-seed-{}/encoder.npy'.format(dataset, i), encoder['centers'])
                np.save('./experiments/{}/dmd-seed-{}/X_train.npy'.format(dataset, i), X_train)
                np.save('./experiments/{}/dmd-seed-{}/y_train.npy'.format(dataset, i), y_train)
                np.save('./experiments/{}/dmd-seed-{}/X_test.npy'.format(dataset, i), X_test)
                np.save('./experiments/{}/dmd-seed-{}/y_test.npy'.format(dataset, i), y_test)
                np.save('./experiments/{}/dmd-seed-{}/train_descrs.npy'.format(dataset, i), train_descrs)
                np.save('./experiments/{}/dmd-seed-{}/test_descrs.npy'.format(dataset, i), test_descrs)

                cls = svm.LinearSVC(penalty='l2', C=0.05)
                print("LinearSVC fitting......")
                cls.fit(train_descrs, y_train)
                # y_train就是标签
                y_predict = cls.predict(test_descrs)
                #
                print("Accuracy Score:{}".format(accuracy_score(y_test, y_predict)))
                print("Precision Score(micro):{}".format(precision_score(y_test, y_predict, average='micro')))
                print("Precision Score(macro):{}".format(precision_score(y_test, y_predict, average='macro')))
                print("Recall Score:(micro){}".format(recall_score(y_test, y_predict, average='micro')))
                print("Recall Score:(macro){}".format(recall_score(y_test, y_predict, average='macro')))
                print("F1 Score:(micro){}".format(f1_score(y_test, y_predict, average='micro')))
                print("F1 Score:(macro){}".format(f1_score(y_test, y_predict, average='macro')))

                print("%s dataset score: %.2f%%" % (dataset, cls.score(test_descrs, y_test) * 100))

                np.save('./experiments/{}/dmd-seed-{}/score.npy'.format(dataset, i), cls.score(test_descrs, y_test))
                np.save('./experiments/{}/dmd-seed-{}/detailed_result.npy'.format(dataset, i),
                        np.concatenate((X_test[:, None], y_test[:, None], y_predict[:, None]), axis=1))

                i += 1
