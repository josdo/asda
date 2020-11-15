## Aerial to Satellite Domain Adaptation on Segmentation Networks

Using adversarial learning and memory regularization to adapt an aerial segmentation network to satellite imagery.

To enable Visdom and Tensorboard for training the aerial segmentation network, enable port forwarding by appending to the SSH login `-- -L 7000:localhost:7000`, then locally opening `localhost:7000`. (You can choose a port besides 7000 as well.)

Adapted from the official implementation of "Unsupervised Scene Adaptation with Memory Regularization in vivo" (IJCAI 2020, IJCV 2020). https://github.com/layumi/Seg-Uncertainty
