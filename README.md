# Object-Detection-Track
A Kaggle competition --Google AI Open Images-Object Detection Track
## Notes
1. It is a very big project. Training with provided Dataset need too much GPU sources. 
2. As there are no so much GPUs, so just use Faster-rcnn model pretrained  by OpenimagesV4 and Yolo3 model pretrained by OpenimagesV4 to do  predictions as a baseline.
3. Faster-rcnn is just for comparing with yolo3 to tune the threshold. It is too slow to complete the test. 
## Usage
1. Each of the 2 files is a Kaggle kernel. Upload pretrained model file by OpenImagesV4 to run it.
2. Top11%(51/454) on Leaderboard.
