# Name
Steve Esguerra
2019-05959

# Keyword Spotting with Transformers
Keyword Spotting app using SPEECHCOMMANDS data set.

# Requirements

To install requirements
```bash
pip install -r requirements.txt
```

# To run 

Clone the repository and then run 
```python
python3 train.py
```
This will automatically download the dataset, train the model, get the most accurate checkpoint, and then convert that checkpoint into torchscript.

To run inferencing mode
```python
python3 kws-infer.py
```
This will use the most accurate .pt which was already provided


# Dataset
Dataset is downloaded from Torch Audio Datasets

```python
from torchaudio.datasets import SPEECHCOMMANDS
```

# Demo

![image](https://user-images.githubusercontent.com/52521318/171010173-ffbf06e3-d754-46e4-9885-3ad4b83deea9.png)

# Labeled Video

[https://drive.google.com/file/d/1oWbnqIOBHS4m-vu3aSMapeOJho90dN0D/view](https://drive.google.com/file/d/1ydyQKPBmYDOYgq3OsWq5yN5BSQ-T1NIn/view?usp=sharing)

# Training Evaluation (Epochs = 20)

```
  | Name    | Type             | Params
---------------------------------------------
0 | encoder | Transformer      | 201 K 
1 | embed   | Linear           | 8.2 K 
2 | fc      | Linear           | 19.0 K
3 | loss    | CrossEntropyLoss | 0     
---------------------------------------------
228 K     Trainable params
0         Non-trainable params
228 K     Total params
0.458     Total estimated model params size (MB)
/usr/local/lib/python3.7/dist-packages/torch/utils/data/dataloader.py:490: UserWarning: This DataLoader will create 6 worker processes in total. Our suggested max number of worker in current system is 2, which is smaller than what this DataLoader is going to create. Please be aware that excessive worker creation might get DataLoader running slow or even freeze, lower the worker number to avoid potential slowness/freeze if necessary.
  cpuset_checked))
/usr/local/lib/python3.7/dist-packages/pytorch_lightning/trainer/connectors/data_connector.py:490: PossibleUserWarning: Your `val_dataloader`'s sampler has shuffling enabled, it is strongly recommended that you turn shuffling off for val/test/predict dataloaders.
  category=PossibleUserWarning,
1558/1558 [02:18<00:00, 11.23it/s, loss=0.388, v_num=11, test_loss=0.548, test_acc=85.30]
Epoch 0, global step 1402: 'test_acc' reached 64.36314 (best 64.36314), saving model to '/content/data/speech_commands/checkpoints/resnet18-kws-best-acc-v2.ckpt' as top 1
Epoch 1, global step 2804: 'test_acc' reached 77.05690 (best 77.05690), saving model to '/content/data/speech_commands/checkpoints/resnet18-kws-best-acc-v2.ckpt' as top 1
Epoch 2, global step 4206: 'test_acc' reached 81.96574 (best 81.96574), saving model to '/content/data/speech_commands/checkpoints/resnet18-kws-best-acc-v2.ckpt' as top 1
Epoch 3, global step 5608: 'test_acc' reached 82.17706 (best 82.17706), saving model to '/content/data/speech_commands/checkpoints/resnet18-kws-best-acc-v2.ckpt' as top 1
Epoch 4, global step 7010: 'test_acc' reached 82.60775 (best 82.60775), saving model to '/content/data/speech_commands/checkpoints/resnet18-kws-best-acc-v2.ckpt' as top 1
Epoch 5, global step 8412: 'test_acc' reached 84.43166 (best 84.43166), saving model to '/content/data/speech_commands/checkpoints/resnet18-kws-best-acc-v2.ckpt' as top 1
Epoch 6, global step 9814: 'test_acc' reached 84.69109 (best 84.69109), saving model to '/content/data/speech_commands/checkpoints/resnet18-kws-best-acc-v2.ckpt' as top 1
Epoch 7, global step 11216: 'test_acc' reached 85.98315 (best 85.98315), saving model to '/content/data/speech_commands/checkpoints/resnet18-kws-best-acc-v2.ckpt' as top 1
Epoch 8, global step 12618: 'test_acc' was not in top 1
Epoch 9, global step 14020: 'test_acc' was not in top 1
Epoch 10, global step 15422: 'test_acc' was not in top 1
Epoch 11, global step 16824: 'test_acc' was not in top 1
Epoch 12, global step 18226: 'test_acc' was not in top 1
Epoch 13, global step 19628: 'test_acc' was not in top 1
Epoch 14, global step 21030: 'test_acc' was not in top 1
LOCAL_RANK: 0 - CUDA_VISIBLE_DEVICES: [0]
/usr/local/lib/python3.7/dist-packages/pytorch_lightning/trainer/connectors/data_connector.py:490: PossibleUserWarning: Your `test_dataloader`'s sampler has shuffling enabled, it is strongly recommended that you turn shuffling off for val/test/predict dataloaders.
  category=PossibleUserWarning,
Testing DataLoader 0: 100%
172/172 [00:10<00:00, 16.84it/s]
────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
       Test metric             DataLoader 0
────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
        test_acc             83.86226654052734
        test_loss           0.5904430150985718
────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
```
 
# References and Acknowledgement
The code is inspired by sample codes from documentation, codes from Kaggle, and Dr. Atienza's Deep Learning Experiments.
