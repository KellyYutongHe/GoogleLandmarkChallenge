import cv2
import numpy as np

img = cv2.imread("example/tokyotower2.jpg")
sift = cv2.xfeatures2d.SIFT_create()
kp, des = sift.detectAndCompute(img,None)
print np.shape(des)
# img=cv2.drawKeypoints(gray,kp,img)
# cv2.imwrite(target + '_keypoints1.jpg',img)

'''
SIFT
'''
sift = cv2.xfeatures2d.SIFT_create()
des_list = []

for image_path in image_paths:
    # print image_path
    img = cv2.imread(image_path)
    sift = cv2.xfeatures2d.SIFT_create()
    kp, des = sift.detectAndCompute(img,None)
    des_list.append(des)
    '''
    Draw Keypoints
    '''
    # img=cv2.drawKeypoints(gray,kp,img)
    # cv2.imwrite(target + '_keypoints1.jpg',img)

descriptors = des_list[0]
for descriptor in des_list[1:]:
    descriptors = np.vstack((descriptors, descriptor))

'''
K-Means
'''
k = 500
centroids, distortion = kmeans(descriptors, k, 1)  # Perform Kmeans with default values

im_features = np.zeros((len(image_paths), k), "float32")
for i in range(len(image_paths)):
    words, distance = vq(des_list[i],centroids)
    for w in words:
        im_features[i][w] += 1

stdSlr = StandardScaler().fit(im_features)
im_features = stdSlr.transform(im_features)

print "K-MEANS Completed"
