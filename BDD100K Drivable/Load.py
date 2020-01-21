from tensorflow.keras.utils import to_categorical
import cv2 as cv
import numpy as np
import glob

ds = 'D:/Dataset/bdd100k/'


def LoadImg(dirs, stop, div, one_hot=False):
    array = []
    for n, dir_ in enumerate(dirs):
        img = cv.imread(dir_, cv.IMREAD_REDUCED_COLOR_4)

        cv.imshow('Load', img)
        cv.waitKey(1)

        if div:
            img = np.array(img) / 255.
        if one_hot:
            img = to_categorical(img, 3)

        array.append(img)

        # 정해진 범위에서 멈춤
        if n + 1 == stop:
            array = np.array(array)
            break

    return array


def Match(dirs, place):
    array = []
    for path in dirs:
        identity = path.split('\\')[-1].split('_')[0] + '.jpg'
        path = ds + 'images/100k/' + place + '/' + identity
        array.append(path)

    return array


def Main():
    trlabel_path = glob.glob(ds + 'drivable_maps/color_labels/train/*.png')
    telabel_path = glob.glob(ds + 'drivable_maps/color_labels/val/*.png')

    np.random.shuffle(trlabel_path)
    # np.random.shuffle(telabel_path)

    trimg_path = Match(trlabel_path, 'train')
    teimg_path = Match(telabel_path, 'val')

    print('Load img..')
    stop = 2000
    trlabel = LoadImg(trlabel_path, stop, True, True)
    trimg = LoadImg(trimg_path, stop, True, False)
    telabel = LoadImg(telabel_path, stop // 5, True, True)
    teimg = LoadImg(teimg_path, stop // 5, True, False)
    cv.destroyAllWindows()
    print('Finished')

    return (trimg, trlabel), (teimg, telabel)
