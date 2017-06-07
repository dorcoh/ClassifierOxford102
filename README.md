# Oxford 102 flowers

repository cloned from https://github.com/jimgoo/caffe-oxford102 , we tried to re-produce his results,

Apparently caffe implementation has changed since then and we couldn't reproduce the AlexNet results (70% on SGD optimizer vs. 93% by jimgoo)

We succeeded in getting the same results for VGG_S, and also improved our trial by 1% to 95.8% by adding one more normalization layer.

In this work, we learned the power of using transfer learning, which can easily benefit the performance of training networks. We did try to train the networks without the pre-trained weights but got poor results.