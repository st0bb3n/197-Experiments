# ObjectDetection-Drinks
FasterRCNN with a MobileNetV3-Large FPN backbone

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

![image](https://user-images.githubusercontent.com/52521318/166231284-2706c11c-e05f-4462-98ad-9c20d7f299bc.png)

# Model Evaualation

Test:  [ 0/59]  eta: 0:00:31  model_time: 0.2838 (0.2838)  evaluator_time: 0.0092 (0.0092)  time: 0.5300  data: 0.2317  max mem: 147

Test:  [58/59]  eta: 0:00:00  model_time: 0.0429 (0.0473)  evaluator_time: 0.0076 (0.0078)  time: 0.0555  data: 0.0053  max mem: 147

Test: Total time: 0:00:03 (0.0671 s / it)

Averaged stats: model_time: 0.0429 (0.0473)  evaluator_time: 0.0076 (0.0078)

Accumulating evaluation results...

DONE (t=0.08s).

IoU metric: bbox

 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.004
 
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.016
 
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.001
 
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = -1.000
 
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.004
 
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.013
 
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.008
 
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.077
 
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.230
 
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = -1.000
 
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.205
 
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.306

# Training Evaluation (Epochs = 5)

insert data
