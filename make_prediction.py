import os
import glob
import cv2
import sys
sys.path.insert(0,'/home/stavsofe@st.technion.ac.il/caffe/python')
import caffe
import lmdb
import numpy as np
from caffe.proto import caffe_pb2

caffe.set_mode_gpu() 

#Size of images
IMAGE_WIDTH = 227
IMAGE_HEIGHT = 227


'''
Image processing helper function
'''
def transform_img(img, img_width=IMAGE_WIDTH, img_height=IMAGE_HEIGHT):

    #Histogram Equalization
    img[:, :, 0] = cv2.equalizeHist(img[:, :, 0])
    img[:, :, 1] = cv2.equalizeHist(img[:, :, 1])
    img[:, :, 2] = cv2.equalizeHist(img[:, :, 2])

    #Image Resizing
    img = cv2.resize(img, (img_width, img_height), interpolation = cv2.INTER_CUBIC)

    return img


'''
Reading mean image, caffe model and its weights 
'''
#Read mean image
mean_blob = caffe_pb2.BlobProto()
with open('/home/stavsofe@st.technion.ac.il/caffe/data/ilsvrc12/imagenet_mean.binaryproto') as f:
    mean_blob.ParseFromString(f.read())
mean_array = np.asarray(mean_blob.data, dtype=np.float32).reshape(
    (mean_blob.channels, mean_blob.height, mean_blob.width))


#Read model architecture and trained model's weights
net = caffe.Net('/home/stavsofe@st.technion.ac.il/caffe-oxford102/AlexNet/deploy.prototxt',
                '/home/stavsofe@st.technion.ac.il/caffe-oxford102/oxford102.caffemodel',
                caffe.TEST)

#Define image transformers
transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
transformer.set_mean('data', mean_array)
transformer.set_transpose('data', (2,0,1))


'''
Making predicitions
'''
##Reading image paths
test_img_paths = [img_path for img_path in glob.glob("/home/stavsofe@st.technion.ac.il/caffe-oxford102/input/*.jpg")]
test_ids = []
preds = []

#Making predictions
for img_path in test_img_paths:
    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    img = transform_img(img, img_width=IMAGE_WIDTH, img_height=IMAGE_HEIGHT)
    
    net.blobs['data'].data[...] = transformer.preprocess('data', img)
    out = net.forward()
    pred_probas = out['prob']

    test_ids = test_ids + [img_path.split('/')[-1][:-4]]
    preds = preds + [pred_probas.argmax()]

    print img_path
    classes =  len(pred_probas[0])
    probsDict = {i:0 for i in range(1,classes+1)}
    for key in probsDict:
        probsDict[key] = pred_probas[0][key-1]
    sort = sorted(probsDict.items(), key=lambda x: x[1], reverse=True)
    for i in range(5):
        print sort[i]
    print pred_probas.argmax()+1
    print '-------'

