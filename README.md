# ObjectDetection-Drinks
FasterRCNN with a MobileNetV3-Large FPN backbone. 

# Notebook Version

Some machines experience an sgmllib error (a module deprecated since python 2.6, removed on 3.0). A Notebook is provided as a workaround (to be opened on Google Colab).

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

Test:  [ 0/59]  eta: 0:00:31  model_time: 0.2838 (0.2838)  evaluator_time: 0.0092 (0.0092)  time: 0.5300  data: 0.2317  max mem: 147 <br>
Test:  [58/59]  eta: 0:00:00  model_time: 0.0429 (0.0473)  evaluator_time: 0.0076 (0.0078)  time: 0.0555  data: 0.0053  max mem: 147 <br>
Test: Total time: 0:00:03 (0.0671 s / it) <br>
Averaged stats: model_time: 0.0429 (0.0473)  evaluator_time: 0.0076 (0.0078) <br>
Accumulating evaluation results... <br>
DONE (t=0.08s). <br>
IoU metric: bbox <br>
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.004 <br>
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.016 <br>
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.001 <br>
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = -1.000 <br>
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.004 <br>
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.013 <br>
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.008 <br>
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.077 <br>
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.230 <br>
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = -1.000 <br>
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.205 <br>
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.306 <br>

# Training Evaluation (Epochs = 10)

Epoch: [9] Total time: 0:00:36 (0.1540 s / it) <br>
creating index... <br>
index created! <br>
Test:  [ 0/59]  eta: 0:00:15  model_time: 0.0744 (0.0744)  evaluator_time: 0.0031 (0.0031)  time: 0.2631  data: 0.1842  max mem: 565 <br>
Test:  [58/59]  eta: 0:00:00  model_time: 0.0326 (0.0347)  evaluator_time: 0.0016 (0.0018)  time: 0.0392  data: 0.0047  max mem: 565 <br>
Test: Total time: 0:00:02 (0.0469 s / it) <br>
Averaged stats: model_time: 0.0326 (0.0347)  evaluator_time: 0.0016 (0.0018) <br>
Accumulating evaluation results... <br>
DONE (t=0.03s). <br>
IoU metric: bbox <br>
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.716 <br>
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.986 <br>
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.897 <br>
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = -1.000 <br>
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.684 <br>
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.798 <br>
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.750 <br>
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.754 <br>
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.754 <br>
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = -1.000 <br>
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.726 <br>
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.824 <br>
