import cv2
import numpy as np
from scipy.cluster.vq import kmeans,vq

class Orb:
    def extract(self, img):
        orb = cv2.ORB_create()
        kp = orb.detect(img, None)
        kp, des = orb.compute(img, kp)
        feature = np.zeros(100, "float32")
        center_path = "data/bovw/center.npy"
        voc = np.load(center_path)
        words, dis = vq(des, voc)
        for w in words:
            feature[w]+=1

        return feature
