import os
import cv2
import time
import numpy as np
from scipy.cluster.vq import *
from sklearn.preprocessing import StandardScaler
from sklearn import svm
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.cluster import MiniBatchKMeans
import pickle



'''
Load Data
'''
# train_path = "trainingdata/"
train_path = "/home/ame/Downloads/dataset-master/JPEGImages/"

image_paths = []
image_classes = []
csv_counter = 0
with open("/home/ame/Downloads/dataset-master/labels.csv", "r") as f:
# with open("train.csv", "r") as f:
    for line in f:
        line = line.split(",")
        if csv_counter == 0:
            csv_counter = 1
            continue
        image_paths.append(train_path + "BloodImage_" + ("%05i" % int(line[0])) + ".jpg")
        # image_paths.append(train_path + line[0][1:-1] + ".jpg")
        # image_classes.append(int(line[2]))
        image_classes.append(line[1])


'''
SIFT & K-Means
'''
sift = cv2.xfeatures2d.SIFT_create()

k = 500
kmeans = MiniBatchKMeans(n_clusters=k, random_state=None)

count = 0
h = open("des.pickle", "wb")
for image_path in image_paths:
    if count % 100 == 0:
        print count, "/", len(image_paths)
    try:
        img = cv2.imread(image_path)
        sift = cv2.xfeatures2d.SIFT_create()
        des = sift.compute(img,None)

        pickle.dump(des, h)

        '''
        Draw Keypoints
        '''
        # kp = sift.detect(img,None)
        # img=cv2.drawKeypoints(gray,kp,img)
        # cv2.imwrite(target + '_keypoints1.jpg',img)

        kmeans.partial_fit(des)
    except cv2.error:
        print "failed: ", count, image_path
    count += 1

h.close()
print "K-MEANS Training Completed"

'''
K-Means
'''
im_features = np.zeros((len(image_paths), k), "float32")
for i in range(len(image_paths)):
    with open("des.pickle", "rb") as h:
        des = pickle.load(h)
        words, distance = vq(des, kmeans.cluster_centers_)
        for w in words:
            im_features[i][w] += 1

stdSlr = StandardScaler().fit(im_features)
im_features = stdSlr.transform(im_features)

with open("vectors_kmeans.pickle", "wb") as h:
    pickle.dump(im_features, h)
print "K-MEANS Vectorizing Completed"

'''
SVM
'''
lin_clf = svm.LinearSVC()
lin_clf.fit(im_features[:-20], image_classes[:-20])
print("SVM Trainning completed")
with open("model_svm.pickle", "wb") as h:
    pickle.dump(lin_clf, h)
print lin_clf.score(im_features[-20:], image_classes[-20:])
result = lin_clf.predict(im_features[-20:])
print(confusion_matrix(image_classes[-20:], result))
print(classification_report(image_classes[-20:], result))