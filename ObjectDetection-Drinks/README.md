# Name
Steve Esguerra
2019-05959

# ObjectDetection-Drinks
FasterRCNN with a MobileNetV3-Large FPN backbone. 

# Notebook Version

It is recommended to use the notebook because I've written this using google colab as I have no GPU. <br>
All of my outputs are generated using google colab including the .pth and evaluation results

# Requirements

To install requirements
```bash
pip install -r requirements.txt
```

# Dataset
Drinks data set in COCO format is used. Data set have a 90/5/5 split. Dataset can be changed by changing 
```python
dataset_path = path/to/dataset/folder
```

# Demo

![image](https://user-images.githubusercontent.com/52521318/166323171-c317cc2a-7710-4611-9553-579da857dc1a.png)

# Labeled Video

https://drive.google.com/file/d/1oWbnqIOBHS4m-vu3aSMapeOJho90dN0D/view

# Model Evaluation

```
Test: [ 0/59] eta: 0:00:09 model_time: 0.0345 (0.0345) evaluator_time: 0.0075 (0.0075) time: 0.1613 data: 0.1180 max mem: 778
Test: [58/59] eta: 0:00:00 model_time: 0.0221 (0.0234) evaluator_time: 0.0059 (0.0064) time: 0.0328 data: 0.0034 max mem: 778
Test: Total time: 0:00:02 (0.0372 s / it)
Averaged stats: model_time: 0.0221 (0.0234) evaluator_time: 0.0059 (0.0064)
Accumulating evaluation results...
DONE (t=0.08s).
IoU metric: bbox
Average Precision (AP) @[ IoU=0.50:0.95 | area= all | maxDets=100 ] = 0.007
Average Precision (AP) @[ IoU=0.50 | area= all | maxDets=100 ] = 0.022
Average Precision (AP) @[ IoU=0.75 | area= all | maxDets=100 ] = 0.001
Average Precision (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.000
Average Precision (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.009
Average Precision (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.008
Average Recall (AR) @[ IoU=0.50:0.95 | area= all | maxDets= 1 ] = 0.008
Average Recall (AR) @[ IoU=0.50:0.95 | area= all | maxDets= 10 ] = 0.092
Average Recall (AR) @[ IoU=0.50:0.95 | area= all | maxDets=100 ] = 0.268
Average Recall (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.000
Average Recall (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.265
Average Recall (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.288
```

# Training (Epoch = 10)

```
Epoch: [9] [ 0/236] eta: 0:01:12 lr: 0.010000 loss: 0.5471 (0.5471) loss_classifier: 0.1453 (0.1453) loss_box_reg: 0.3979 (0.3979) loss_objectness: 0.0032 (0.0032) loss_rpn_box_reg: 0.0007 (0.0007) time: 0.3056 data: 0.2130 max mem: 793
Epoch: [9] [ 50/236] eta: 0:00:16 lr: 0.010000 loss: 0.4727 (0.5048) loss_classifier: 0.1253 (0.1343) loss_box_reg: 0.3351 (0.3651) loss_objectness: 0.0032 (0.0040) loss_rpn_box_reg: 0.0013 (0.0015) time: 0.0822 data: 0.0136 max mem: 793
Epoch: [9] [100/236] eta: 0:00:11 lr: 0.010000 loss: 0.4957 (0.5203) loss_classifier: 0.1255 (0.1312) loss_box_reg: 0.3621 (0.3841) loss_objectness: 0.0029 (0.0035) loss_rpn_box_reg: 0.0015 (0.0015) time: 0.0820 data: 0.0132 max mem: 793
Epoch: [9] [150/236] eta: 0:00:07 lr: 0.010000 loss: 0.5244 (0.5222) loss_classifier: 0.1189 (0.1339) loss_box_reg: 0.4097 (0.3832) loss_objectness: 0.0029 (0.0034) loss_rpn_box_reg: 0.0015 (0.0016) time: 0.0836 data: 0.0144 max mem: 793
Epoch: [9] [200/236] eta: 0:00:03 lr: 0.010000 loss: 0.5424 (0.5249) loss_classifier: 0.1318 (0.1358) loss_box_reg: 0.4170 (0.3837) loss_objectness: 0.0034 (0.0037) loss_rpn_box_reg: 0.0014 (0.0016) time: 0.0835 data: 0.0137 max mem: 793
Epoch: [9] [235/236] eta: 0:00:00 lr: 0.010000 loss: 0.5015 (0.5252) loss_classifier: 0.1380 (0.1360) loss_box_reg: 0.3474 (0.3838) loss_objectness: 0.0025 (0.0037) loss_rpn_box_reg: 0.0014 (0.0016) time: 0.0776 data: 0.0133 max mem: 793
Epoch: [9] Total time: 0:00:19 (0.0836 s / it)
creating index...
index created!
Test: [ 0/59] eta: 0:00:08 model_time: 0.0288 (0.0288) evaluator_time: 0.0022 (0.0022) time: 0.1489 data: 0.1169 max mem: 793
Test: [58/59] eta: 0:00:00 model_time: 0.0202 (0.0210) evaluator_time: 0.0015 (0.0015) time: 0.0269 data: 0.0043 max mem: 793
Test: Total time: 0:00:01 (0.0301 s / it)
Averaged stats: model_time: 0.0202 (0.0210) evaluator_time: 0.0015 (0.0015)
Accumulating evaluation results...
DONE (t=0.03s).
IoU metric: bbox
Average Precision (AP) @[ IoU=0.50:0.95 | area= all | maxDets=100 ] = 0.792
Average Precision (AP) @[ IoU=0.50 | area= all | maxDets=100 ] = 1.000
Average Precision (AP) @[ IoU=0.75 | area= all | maxDets=100 ] = 0.973
Average Precision (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.400
Average Precision (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.781
Average Precision (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.854
Average Recall (AR) @[ IoU=0.50:0.95 | area= all | maxDets= 1 ] = 0.835
Average Recall (AR) @[ IoU=0.50:0.95 | area= all | maxDets= 10 ] = 0.836
Average Recall (AR) @[ IoU=0.50:0.95 | area= all | maxDets=100 ] = 0.836
Average Recall (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.700
Average Recall (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.822
Average Recall (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.888
```
 # References and Acknowledgement
 The code is inspired by sample codes from documentation, codes from Kaggle, and Dr. Atienza's Deep Learning Experiments.
