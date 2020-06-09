import numpy as np
import cv2
def coordianteFelt(imageList,pts1,pts2):
    x_c=np.zeros((2,80))
    y_c=np.zeros((2,80))
    block_radius=7
    max_scale= 4
    len_img=len(imageList)
    g_stor1=np.zeros((len_img,80))
    g_stor2=np.zeros((len_img,80))
    i1 = 0
    pts1[pts1 > block_radius - max_scale + 1] = block_radius - max_scale + 1
    pts1[pts1 < -block_radius] = -block_radius
    pts2[pts2 > block_radius - max_scale + 1] = block_radius - max_scale + 1
    pts2[pts2 < -block_radius] = -block_radius
    # 把坐标都挪到图片里面
    pts1 = (pts1 + block_radius + 1).astype(np.uint16)
    pts2 = (pts2 + block_radius + 1).astype(np.uint16)
    for image in imageList:

        img=cv2.imread(image,cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (200, 200))
        x = cv2.Sobel(img, cv2.CV_16S, 1, 0)
        y = cv2.Sobel(img, cv2.CV_16S, 0, 1)
        absX = cv2.convertScaleAbs(x)  # 转回uint8
        absY = cv2.convertScaleAbs(y)
        dst = cv2.addWeighted(absX, 0.5, absY, 0.5, 0)

        for i in range(8):
            a1 = pts1[:, i * 10:(i + 1) * 10]
            a2 = pts2[:, i * 10:(i + 1) * 10]
            s1 = np.zeros(10)
            s2 = np.zeros(10)
            for j in range(10):
                sob_1 = dst[a1[0, j], a1[1, j]]
                sob_2 = dst[a2[0, j], a2[1, j]]
                s1[j] = sob_1
                s2[j] = sob_2
            g_stor1[i1,i*10:(i+1)*10]=s1
            g_stor2[i1,i*10:(i+1)*10]=s2
        i1=i1+1
    sum_g1=np.sum(g_stor1,axis=0)
    sum_g2=np.sum(g_stor2,axis=0)
    for i in range(8):
        s_c1=sum_g1[i: :8]
        s_c2=sum_g2[i: :8]
        s_g1_sort = np.argsort(-s_c1)
        s_g2_sort = np.argsort(-s_c2)
        for j in range(10):
            x_c[:,(j*8)+i]=pts1[:,(s_g1_sort[j]*8)+i]
            y_c[:,(j*8)+i]=pts2[:,(s_g2_sort[j]*8)+i]
    x_c=x_c[:, : :2]
    y_c=y_c[:, : :2]
    x_c=x_c.astype(int)
    y_c=y_c.astype(int)
    return x_c,y_c