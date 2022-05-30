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

# Demo Video

[https://drive.google.com/file/d/1oWbnqIOBHS4m-vu3aSMapeOJho90dN0D/view](https://drive.google.com/file/d/1ydyQKPBmYDOYgq3OsWq5yN5BSQ-T1NIn/view?usp=sharing)

# Training Evaluation (Epochs = 20)

```
  | Name    | Type             | Params
---------------------------------------------
0 | encoder | Transformer      | 597 K 
1 | embed   | Linear           | 16.4 K
2 | fc      | Linear           | 37.9 K
3 | loss    | CrossEntropyLoss | 0     
---------------------------------------------
651 K     Trainable params
0         Non-trainable params
651 K     Total params
1.304     Total estimated model params size (MB)
Sanity Checking DataLoader 0: 100%
2/2 [00:00<00:00, 7.02it/s]
Epoch 19: 100%
1558/1558 [02:11<00:00, 11.86it/s, loss=0.204, v_num=0, test_loss=0.397, test_acc=90.90]
Validation DataLoader 0: 100%
156/156 [00:09<00:00, 18.96it/s]
Epoch 0, global step 1402: 'test_acc' reached 81.72289 (best 81.72289), saving model to '/content/data/speech_commands/checkpoints/resnet18-kws-best-acc.ckpt' as top 1
Validation DataLoader 0: 100%
156/156 [00:08<00:00, 20.35it/s]
Epoch 1, global step 2804: 'test_acc' reached 85.20387 (best 85.20387), saving model to '/content/data/speech_commands/checkpoints/resnet18-kws-best-acc.ckpt' as top 1
Validation: 0it [00:00, ?it/s]
Epoch 2, global step 4206: 'test_acc' reached 85.88152 (best 85.88152), saving model to '/content/data/speech_commands/checkpoints/resnet18-kws-best-acc.ckpt' as top 1
Epoch 3, global step 5608: 'test_acc' reached 89.26840 (best 89.26840), saving model to '/content/data/speech_commands/checkpoints/resnet18-kws-best-acc.ckpt' as top 1
Epoch 4, global step 7010: 'test_acc' was not in top 1
Epoch 5, global step 8412: 'test_acc' was not in top 1
Epoch 6, global step 9814: 'test_acc' was not in top 1
Epoch 7, global step 11216: 'test_acc' reached 90.37213 (best 90.37213), saving model to '/content/data/speech_commands/checkpoints/resnet18-kws-best-acc.ckpt' as top 1
Epoch 8, global step 12618: 'test_acc' was not in top 1
Epoch 9, global step 14020: 'test_acc' was not in top 1
Epoch 10, global step 15422: 'test_acc' was not in top 1
Epoch 11, global step 16824: 'test_acc' was not in top 1
Epoch 12, global step 18226: 'test_acc' reached 90.47131 (best 90.47131), saving model to '/content/data/speech_commands/checkpoints/resnet18-kws-best-acc.ckpt' as top 1
Epoch 13, global step 19628: 'test_acc' was not in top 1
Epoch 14, global step 21030: 'test_acc' was not in top 1
Epoch 15, global step 22432: 'test_acc' reached 91.62414 (best 91.62414), saving model to '/content/data/speech_commands/checkpoints/resnet18-kws-best-acc.ckpt' as top 1
Epoch 16, global step 23834: 'test_acc' was not in top 1
Epoch 17, global step 25236: 'test_acc' was not in top 1
Epoch 18, global step 26638: 'test_acc' was not in top 1
Epoch 19, global step 28040: 'test_acc' was not in top 1
LOCAL_RANK: 0 - CUDA_VISIBLE_DEVICES: [0]
Testing DataLoader 0: 100%
172/172 [00:10<00:00, 17.88it/s]
────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
       Test metric             DataLoader 0
────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
        test_acc             90.2952880859375
        test_loss           0.41182997822761536
────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
```
 
# References and Acknowledgement
The code is inspired by sample codes from documentation, codes from Kaggle, and Dr. Atienza's Deep Learning Experiments.
