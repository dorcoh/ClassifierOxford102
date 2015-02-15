# caffe-oxford102
This is for training a deep convolutional neural network to classify images in the Oxford 102 category flower dataset. The model is fine-tuned from the Caffe reference model (AlexNet trained on ILSVRC 2012).

Download the Oxford 102 category dataset:

`./get_oxford102.sh`

Create the Caffe style training and testing set files:

`./create_caffe_splits.py`
