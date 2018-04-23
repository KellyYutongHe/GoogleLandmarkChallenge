import os
import cv2
import time
import numpy as np
from scipy.cluster.vq import vq
from sklearn.preprocessing import StandardScaler
from sklearn import svm
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.cluster import MiniBatchKMeans
import pickle
from sklearn import linear_model



'''
Load Data
'''
train_path = "trainingdata/"
# train_path = "/home/ame/Downloads/dataset-master/JPEGImages/"

image_paths = []
image_classes = []
csv_counter = 0
# with open("/home/ame/Downloads/dataset-master/labels.csv", "r") as f:
with open("train.csv", "r") as f:
    for line in f:
        line = line.split(",")
        if csv_counter == 0:
            csv_counter = 1
            continue
        # image_paths.append(train_path + "BloodImage_" + ("%05i" % int(line[0])) + ".jpg")
        image_paths.append(train_path + line[0][1:-1] + ".jpg")
        image_classes.append(int(line[2]))
        # image_classes.append(line[1])


'''
SIFT & K-Means
'''
sift = cv2.xfeatures2d.SIFT_create()

k = 250
count = 0
batch_size = 100
test_split = 100
des_list = []


if os.path.exists("clusters.pickle") and os.path.exists("model_svm.pickle"):
    with open("model_svm.pickle", "rb") as v, open("clusters.pickle", "rb") as t:
        print "lol"
        lin_clf = pickle.load(v)
        kmeans = pickle.load(t)
else:
    lin_clf = linear_model.SGDClassifier()
    kmeans = MiniBatchKMeans(n_clusters=k, random_state=None)

t0 = time.time()
# descriptors = np.array([], dtype=np.int32).reshape(0,128)
im_features = np.zeros((batch_size, k), "float32")
im_classes = image_classes[:batch_size]
batch_count = 0
for image_path in image_paths:
    if count % 100 == 0:
        print count, "/", len(image_paths)

    try:
        img = cv2.imread(image_path)
        sift = cv2.xfeatures2d.SIFT_create()
        kp, des = sift.detectAndCompute(img,None)
        des_list.append(des)
        # descriptors = np.vstack((descriptors, des))
        kmeans.partial_fit(des)
        batch_count += 1

        '''
        Draw Keypoints
        '''
        # img=cv2.drawKeypoints(gray,kp,img)
        # cv2.imwrite(target + '_keypoints1.jpg',img)

        if (count % batch_size == 0 or len(image_paths) - count  < 3):
            print "saving:", count, "/", len(image_paths)
            '''
            K-Means
            '''
            # kmeans.partial_fit(descriptors)
            i = 0
            for des in des_list:
                words, distance = vq(des, kmeans.cluster_centers_)
                for w in words:
                    im_features[i][w] += 1
                i += 1
            des_list = []
            # descriptors = np.array([], dtype=np.int32).reshape(0,128)
            print "K-MEANS Partial Completed", str(time.time() - t0)

            '''
            SVM
            '''
            lin_clf.fit(im_features, im_classes)
            with open("model_svm.pickle", "wb") as f1, open("clusters.pickle", "wb") as f2:
                pickle.dump(lin_clf, f1)
                pickle.dump(kmeans, f2)
            if count % test_split == 0:
                print "SCORE:", lin_clf.score(im_features, im_classes)
            if len(image_paths) - count - 1 < test_split:
                break
            im_features = np.zeros((batch_size, k), "float32")
            im_classes = image_classes[count + 1: count + batch_size + 1]
            batch_count = 0
            print "SVM Trainning completed", str(time.time() - t0)

    except cv2.error:
        print "failed: ", count, image_path
        im_features = np.delete(im_features, (batch_count), axis=0)
        im_classes = np.delete(im_classes, (batch_count), axis=0)

    count += 1

dt = time.time() - t0
print('done in %.2fs.' % dt)

# for j in range(skips, len(image_paths)):
#     im_features = np.delete(im_features, (j), axis=0)

# stdSlr = StandardScaler().fit(im_features)
# im_features = stdSlr.transform(im_features)
#
# with open("vectors_kmeans.pickle", "wb") as h:
#     pickle.dump(im_features, h)
# print "K-MEANS Vectorizing Completed"

'''
SVM
'''
# lin_clf = svm.LinearSVC()
# lin_clf.fit(im_features[:-test_split], image_classes[:-test_split])
# print("SVM Trainning completed")
# with open("model_svm.pickle", "wb") as h:
#     pickle.dump(lin_clf, h)
# print lin_clf.score(im_features[-test_split:], image_classes[-test_split:])
# result = lin_clf.predict(im_features[-test_split:])
# print(confusion_matrix(image_classes[-test_split:], result))
# print(classification_report(image_classes[-test_split:], result))

'''
Testing
'''
im_features = np.zeros((test_split, k), "float32")
im_classes = image_classes[-test_split:]
count = 0
score = 0
result = []

with open("model_svm.pickle", "rb") as v, open("clusters.pickle", "rb") as t:
    lin_clf = pickle.load(v)
    kmeans = pickle.load(t)

t0 = time.time()
for image_path in image_paths[-test_split:]:
    try:
        img = cv2.imread(image_path)
        sift = cv2.xfeatures2d.SIFT_create()
        kp, des = sift.detectAndCompute(img,None)
        des_list.append(des)

        '''
        Draw Keypoints
        '''
        # img=cv2.drawKeypoints(gray,kp,img)
        # cv2.imwrite(target + '_keypoints1.jpg',img)

        '''
        K-Means
        '''
        i = 0
        for des in des_list:
            words, distance = vq(des, kmeans.cluster_centers_)
            for w in words:
                im_features[i][w] += 1
            i += 1
        des_list = []
    except cv2.error:
        print "failed: ", count, image_path
        im_features = np.delete(im_features, (count), axis=0)
        im_classes = np.delete(im_classes, (count), axis=0)

    count += 1

'''
SVM
'''
print "TESTING SCORE:", lin_clf.score(im_features, im_classes)
result = lin_clf.predict(im_features)
# print(confusion_matrix(im_classes, result))
print(classification_report(im_classes, result))
