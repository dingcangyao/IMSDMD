import numpy as np
import os


def get_number_of_class(dataset):
    datasets = {'curet': 205, 'KTH_TIPS': 81, 'umd':40, 'fmd': 100, 'uiuc': 40}
    if dataset not in datasets.keys():
        print("{} is not exist".format(dataset))
        return None
    else:
        return datasets[dataset]


def get_images(dataset):
    datasets = {'curet': './data/curet', 'KTH_TIPS': './data/KTH_TIPS', 'umd': './data/umd',
                'uiuc': './data/UIUC'}
    if dataset not in datasets.keys():#如果数据库不存在，就返回没有这个数据库
        print("{} is not exist".format(dataset))
        return None
    else:
        class_name_list = os.listdir(datasets[dataset])
        class_path_list = [os.path.join(datasets[dataset].replace('\\', '/'), class_name) for class_name in
                           class_name_list]
        image_path_list = []
        label_list = []
        for class_path in class_path_list:
            image_name_per_class_list = os.listdir(class_path)
            image_path_per_class_list = [os.path.join(class_path, image_name).replace('\\', '/') for image_name in
                                         image_name_per_class_list]
            image_path_list.append(image_path_per_class_list)
            # s=[os.path.split(class_path)[1] for i in range(get_number_of_class(dataset))]#把路径与名字分开，然后返回一个数组
            # print('nishaksahska:'+s)
            label_list.append([os.path.split(class_path)[1] for i in range(get_number_of_class(dataset))])
            # print(label_list)
        if(dataset=='umd' or dataset=='curet'):
            a1 = len(image_path_list)

            c = np.array(image_path_list[0])
            for i in range(a1-1):
                c1 = np.array(image_path_list[i+1])
                c=np.hstack((c,c1))
        else:
            c=np.array(image_path_list).flatten()
        b=np.array(label_list).flatten()
        return c,b#返回图像地址还有对应的标签列表
