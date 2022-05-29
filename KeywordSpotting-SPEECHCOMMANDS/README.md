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

# Dataset
Dataset is downloaded from Torch Audio Datasets

```python
from torchaudio.datasets import SPEECHCOMMANDS
```

# Demo

![image](https://user-images.githubusercontent.com/52521318/166323171-c317cc2a-7710-4611-9553-579da857dc1a.png)

# Labeled Video

https://drive.google.com/file/d/1oWbnqIOBHS4m-vu3aSMapeOJho90dN0D/view

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
Sanity Checking: 0it [00:00, ?it/s]/home/esguerra_s/.local/lib/python3.8/site-packages/pytorch_lightning/trainer/connectors/data_connector.py:487: 
Epoch 0: 100%|████████████████| 3115/3115 [02:22<00:00, 21.79it/s, loss=2.06, v_num=14, test_loss=1.430, test_acc=64.00]Epoch 0, global step 2803: 'test_acc' reached 64.04109 (best 64.04109), saving model to '/home/esguerra_s/data/speech_commands/checkpoints/resnet18-kws-best-acc.ckpt' as top 1
Epoch 1: 100%|████████████████| 3115/3115 [02:22<00:00, 21.80it/s, loss=1.06, v_num=14, test_loss=0.826, test_acc=76.30]Epoch 1, global step 5606: 'test_acc' reached 76.30485 (best 76.30485), saving model to '/home/esguerra_s/data/speech_commands/checkpoints/resnet18-kws-best-acc.ckpt' as top 1
Epoch 2: 100%|███████████████| 3115/3115 [02:23<00:00, 21.77it/s, loss=0.741, v_num=14, test_loss=0.779, test_acc=77.50]Epoch 2, global step 8409: 'test_acc' reached 77.45669 (best 77.45669), saving model to '/home/esguerra_s/data/speech_commands/checkpoints/resnet18-kws-best-acc.ckpt' as top 1
Epoch 3: 100%|███████████████| 3115/3115 [02:23<00:00, 21.75it/s, loss=0.719, v_num=14, test_loss=0.666, test_acc=81.40]Epoch 3, global step 11212: 'test_acc' reached 81.42407 (best 81.42407), saving model to '/home/esguerra_s/data/speech_commands/checkpoints/resnet18-kws-best-acc.ckpt' as top 1
Epoch 4: 100%|███████████████| 3115/3115 [02:23<00:00, 21.78it/s, loss=0.712, v_num=14, test_loss=0.688, test_acc=81.10]Epoch 4, global step 14015: 'test_acc' was not in top 1
Epoch 5: 100%|███████████████| 3115/3115 [02:23<00:00, 21.75it/s, loss=0.778, v_num=14, test_loss=0.903, test_acc=76.80]Epoch 5, global step 16818: 'test_acc' was not in top 1
Epoch 6: 100%|███████████████| 3115/3115 [02:23<00:00, 21.77it/s, loss=0.653, v_num=14, test_loss=0.574, test_acc=84.50]Epoch 6, global step 19621: 'test_acc' reached 84.47205 (best 84.47205), saving model to '/home/esguerra_s/data/speech_commands/checkpoints/resnet18-kws-best-acc.ckpt' as top 1
Epoch 7: 100%|████████████████| 3115/3115 [02:22<00:00, 21.79it/s, loss=7.24, v_num=14, test_loss=8.930, test_acc=54.30]Epoch 7, global step 22424: 'test_acc' was not in top 1
Epoch 8: 100%|████████████████| 3115/3115 [02:23<00:00, 21.76it/s, loss=1.67, v_num=14, test_loss=2.170, test_acc=73.40]Epoch 8, global step 25227: 'test_acc' was not in top 1
Epoch 9: 100%|████████████████| 3115/3115 [02:23<00:00, 21.75it/s, loss=2.02, v_num=14, test_loss=3.100, test_acc=64.20]Epoch 9, global step 28030: 'test_acc' was not in top 1
Epoch 10: 100%|██████████████| 3115/3115 [02:23<00:00, 21.73it/s, loss=0.387, v_num=14, test_loss=0.491, test_acc=87.10]Epoch 10, global step 30833: 'test_acc' reached 87.08520 (best 87.08520), saving model to '/home/esguerra_s/data/speech_commands/checkpoints/resnet18-kws-best-acc.ckpt' as top 1
Epoch 11: 100%|██████████████| 3115/3115 [02:23<00:00, 21.76it/s, loss=0.569, v_num=14, test_loss=0.640, test_acc=82.10]Epoch 11, global step 33636: 'test_acc' was not in top 1
Epoch 12: 100%|██████████████| 3115/3115 [02:23<00:00, 21.75it/s, loss=0.728, v_num=14, test_loss=0.855, test_acc=81.80]Epoch 12, global step 36439: 'test_acc' was not in top 1
Epoch 13: 100%|██████████████| 3115/3115 [02:23<00:00, 21.75it/s, loss=0.483, v_num=14, test_loss=0.509, test_acc=86.70]Epoch 13, global step 39242: 'test_acc' was not in top 1
Epoch 14: 100%|██████████████| 3115/3115 [02:23<00:00, 21.76it/s, loss=0.378, v_num=14, test_loss=0.691, test_acc=82.60]Epoch 14, global step 42045: 'test_acc' was not in top 1
Epoch 15: 100%|███████████████| 3115/3115 [02:23<00:00, 21.75it/s, loss=0.31, v_num=14, test_loss=0.536, test_acc=86.60]Epoch 15, global step 44848: 'test_acc' was not in top 1
Epoch 16: 100%|██████████████| 3115/3115 [02:23<00:00, 21.73it/s, loss=0.306, v_num=14, test_loss=0.552, test_acc=86.20]Epoch 16, global step 47651: 'test_acc' was not in top 1
Epoch 17: 100%|██████████████| 3115/3115 [02:23<00:00, 21.75it/s, loss=0.273, v_num=14, test_loss=0.509, test_acc=87.50]Epoch 17, global step 50454: 'test_acc' reached 87.45786 (best 87.45786), saving model to '/home/esguerra_s/data/speech_commands/checkpoints/resnet18-kws-best-acc.ckpt' as top 1
Epoch 18: 100%|██████████████| 3115/3115 [02:23<00:00, 21.75it/s, loss=0.267, v_num=14, test_loss=0.497, test_acc=87.60]Epoch 18, global step 53257: 'test_acc' reached 87.64816 (best 87.64816), saving model to '/home/esguerra_s/data/speech_commands/checkpoints/resnet18-kws-best-acc.ckpt' as top 1
Epoch 19: 100%|██████████████| 3115/3115 [02:23<00:00, 21.78it/s, loss=0.246, v_num=14, test_loss=0.512, test_acc=87.30]
Epoch 19, global step 56060: 'test_acc' was not in top 1
Epoch 19: 100%|██████████████| 3115/3115 [02:23<00:00, 21.78it/s, loss=0.246, v_num=14, test_loss=0.512, test_acc=87.30]
LOCAL_RANK: 0 - CUDA_VISIBLE_DEVICES: [0]
Testing DataLoader 0: 100%|███████████████████████████████████████████████████████████| 344/344 [00:05<00:00, 68.47it/s]
────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
       Test metric             DataLoader 0
────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
        test_acc             85.00713348388672
        test_loss            0.580515444278717
────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
```
 
# References and Acknowledgement
The code is inspired by sample codes from documentation, codes from Kaggle, and Dr. Atienza's Deep Learning Experiments.
