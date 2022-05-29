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
Sanity Checking: 0it [00:00, ?it/s]/home/esguerra_s/.local/lib/python3.8/site-packages/pytorch_lightning/trainer/connectors/data_connector.py:487: PossibleUserWarning: Your `val_dataloader`'s sampler has shuffling enabled, it is strongly recommended that you turn shuffling off for val/test/predict dataloaders.
  rank_zero_warn(
Epoch 0: 100%|████████████████| 1558/1558 [01:13<00:00, 21.21it/s, loss=1.22, v_num=21, test_loss=1.140, test_acc=66.70]Epoch 0, global step 1402: 'test_acc' reached 66.71741 (best 66.71741), saving model to '/home/esguerra_s/data/speech_commands/checkpoints/resnet18-kws-best-acc.pt-v2.ckpt' as top 1
Epoch 1: 100%|███████████████| 1558/1558 [01:13<00:00, 21.17it/s, loss=0.905, v_num=21, test_loss=0.752, test_acc=77.60]Epoch 1, global step 2804: 'test_acc' reached 77.60828 (best 77.60828), saving model to '/home/esguerra_s/data/speech_commands/checkpoints/resnet18-kws-best-acc.pt-v2.ckpt' as top 1
Epoch 2: 100%|███████████████| 1558/1558 [01:13<00:00, 21.20it/s, loss=0.775, v_num=21, test_loss=0.625, test_acc=82.00]Epoch 2, global step 4206: 'test_acc' reached 81.95523 (best 81.95523), saving model to '/home/esguerra_s/data/speech_commands/checkpoints/resnet18-kws-best-acc.pt-v2.ckpt' as top 1
Epoch 3: 100%|███████████████| 1558/1558 [01:13<00:00, 21.15it/s, loss=0.675, v_num=21, test_loss=0.595, test_acc=82.40]Epoch 3, global step 5608: 'test_acc' reached 82.44404 (best 82.44404), saving model to '/home/esguerra_s/data/speech_commands/checkpoints/resnet18-kws-best-acc.pt-v2.ckpt' as top 1
Epoch 4: 100%|███████████████| 1558/1558 [01:13<00:00, 21.19it/s, loss=0.865, v_num=21, test_loss=0.735, test_acc=81.00]Epoch 4, global step 7010: 'test_acc' was not in top 1
Epoch 5: 100%|████████████████| 1558/1558 [01:13<00:00, 21.15it/s, loss=0.56, v_num=21, test_loss=0.541, test_acc=84.10]Epoch 5, global step 8412: 'test_acc' reached 84.11163 (best 84.11163), saving model to '/home/esguerra_s/data/speech_commands/checkpoints/resnet18-kws-best-acc.pt-v2.ckpt' as top 1
Epoch 6: 100%|███████████████| 1558/1558 [01:13<00:00, 21.14it/s, loss=0.625, v_num=21, test_loss=0.520, test_acc=85.10]Epoch 6, global step 9814: 'test_acc' reached 85.06068 (best 85.06068), saving model to '/home/esguerra_s/data/speech_commands/checkpoints/resnet18-kws-best-acc.pt-v2.ckpt' as top 1
Epoch 7: 100%|███████████████| 1558/1558 [01:13<00:00, 21.16it/s, loss=0.681, v_num=21, test_loss=0.682, test_acc=82.70]Epoch 7, global step 11216: 'test_acc' was not in top 1
Epoch 8: 100%|███████████████| 1558/1558 [01:13<00:00, 21.15it/s, loss=0.476, v_num=21, test_loss=0.525, test_acc=85.20]Epoch 8, global step 12618: 'test_acc' reached 85.18942 (best 85.18942), saving model to '/home/esguerra_s/data/speech_commands/checkpoints/resnet18-kws-best-acc.pt-v2.ckpt' as top 1
Epoch 9: 100%|███████████████| 1558/1558 [01:13<00:00, 21.16it/s, loss=0.626, v_num=21, test_loss=0.607, test_acc=83.70]Epoch 9, global step 14020: 'test_acc' was not in top 1
Epoch 10: 100%|██████████████| 1558/1558 [01:13<00:00, 21.15it/s, loss=0.583, v_num=21, test_loss=0.649, test_acc=83.50]Epoch 10, global step 15422: 'test_acc' was not in top 1
Epoch 11: 100%|██████████████| 1558/1558 [01:13<00:00, 21.16it/s, loss=0.383, v_num=21, test_loss=0.544, test_acc=85.60]Epoch 11, global step 16824: 'test_acc' reached 85.63358 (best 85.63358), saving model to '/home/esguerra_s/data/speech_commands/checkpoints/resnet18-kws-best-acc.pt-v2.ckpt' as top 1
Epoch 12: 100%|██████████████| 1558/1558 [01:13<00:00, 21.14it/s, loss=0.458, v_num=21, test_loss=0.787, test_acc=80.20]Epoch 12, global step 18226: 'test_acc' was not in top 1
Epoch 13: 100%|███████████████| 1558/1558 [01:13<00:00, 21.17it/s, loss=45.1, v_num=21, test_loss=3.900, test_acc=71.90]Epoch 13, global step 19628: 'test_acc' was not in top 1
Epoch 14: 100%|███████████████| 1558/1558 [01:13<00:00, 21.13it/s, loss=0.26, v_num=21, test_loss=0.546, test_acc=85.50]Epoch 14, global step 21030: 'test_acc' was not in top 1
Epoch 15: 100%|██████████████| 1558/1558 [01:13<00:00, 21.15it/s, loss=0.275, v_num=21, test_loss=0.499, test_acc=87.10]Epoch 15, global step 22432: 'test_acc' reached 87.11545 (best 87.11545), saving model to '/home/esguerra_s/data/speech_commands/checkpoints/resnet18-kws-best-acc.pt-v2.ckpt' as top 1
Epoch 16: 100%|██████████████| 1558/1558 [01:13<00:00, 21.15it/s, loss=0.263, v_num=21, test_loss=0.531, test_acc=86.50]Epoch 16, global step 23834: 'test_acc' was not in top 1
Epoch 17: 100%|██████████████| 1558/1558 [01:13<00:00, 21.14it/s, loss=0.262, v_num=21, test_loss=0.532, test_acc=86.60]Epoch 17, global step 25236: 'test_acc' was not in top 1
Epoch 18: 100%|██████████████| 1558/1558 [01:13<00:00, 21.13it/s, loss=0.276, v_num=21, test_loss=0.591, test_acc=84.70]Epoch 18, global step 26638: 'test_acc' was not in top 1
Epoch 19: 100%|███████████████| 1558/1558 [01:13<00:00, 21.14it/s, loss=0.24, v_num=21, test_loss=0.554, test_acc=85.40]Epoch 19, global step 28040: 'test_acc' was not in top 1
Epoch 20: 100%|██████████████| 1558/1558 [01:13<00:00, 21.15it/s, loss=0.243, v_num=21, test_loss=0.511, test_acc=87.00]Epoch 20, global step 29442: 'test_acc' was not in top 1
Epoch 21: 100%|██████████████| 1558/1558 [01:13<00:00, 21.13it/s, loss=0.206, v_num=21, test_loss=0.515, test_acc=86.60]Epoch 21, global step 30844: 'test_acc' was not in top 1
Epoch 22: 100%|██████████████| 1558/1558 [01:13<00:00, 21.16it/s, loss=0.242, v_num=21, test_loss=0.534, test_acc=85.70]Epoch 22, global step 32246: 'test_acc' was not in top 1
Epoch 23: 100%|██████████████| 1558/1558 [01:13<00:00, 21.14it/s, loss=0.143, v_num=21, test_loss=0.489, test_acc=87.90]Epoch 23, global step 33648: 'test_acc' reached 87.92429 (best 87.92429), saving model to '/home/esguerra_s/data/speech_commands/checkpoints/resnet18-kws-best-acc.pt-v2.ckpt' as top 1
Epoch 24: 100%|██████████████| 1558/1558 [01:13<00:00, 21.15it/s, loss=0.219, v_num=21, test_loss=0.536, test_acc=86.10]Epoch 24, global step 35050: 'test_acc' was not in top 1
Epoch 25: 100%|██████████████| 1558/1558 [01:13<00:00, 21.17it/s, loss=0.134, v_num=21, test_loss=0.515, test_acc=86.70]Epoch 25, global step 36452: 'test_acc' was not in top 1
Epoch 26: 100%|██████████████| 1558/1558 [01:13<00:00, 21.16it/s, loss=0.173, v_num=21, test_loss=0.527, test_acc=86.20]Epoch 26, global step 37854: 'test_acc' was not in top 1
Epoch 27: 100%|███████████████| 1558/1558 [01:13<00:00, 21.15it/s, loss=0.14, v_num=21, test_loss=0.530, test_acc=86.20]Epoch 27, global step 39256: 'test_acc' was not in top 1
Epoch 28: 100%|█████████████| 1558/1558 [01:13<00:00, 21.16it/s, loss=0.0952, v_num=21, test_loss=0.525, test_acc=86.40]Epoch 28, global step 40658: 'test_acc' was not in top 1
Epoch 29: 100%|███████████████| 1558/1558 [01:13<00:00, 21.16it/s, loss=0.15, v_num=21, test_loss=0.534, test_acc=86.00]Epoch 29, global step 42060: 'test_acc' was not in top 1
Epoch 29: 100%|███████████████| 1558/1558 [01:13<00:00, 21.16it/s, loss=0.15, v_num=21, test_loss=0.534, test_acc=86.00]LOCAL_RANK: 0 - CUDA_VISIBLE_DEVICES: [0]
/home/esguerra_s/.local/lib/python3.8/site-packages/pytorch_lightning/trainer/connectors/data_connector.py:487: PossibleUserWarning: Your `test_dataloader`'s sampler has shuffling enabled, it is strongly recommended that you turn shuffling off for val/test/predict dataloaders.
  rank_zero_warn(
Testing DataLoader 0: 100%|███████████████████████████████████████████████████████████| 172/172 [00:03<00:00, 49.19it/s]────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────       Test metric             DataLoader 0
────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────        test_acc             84.38958740234375
        test_loss           0.5757566094398499
────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
```
 
# References and Acknowledgement
The code is inspired by sample codes from documentation, codes from Kaggle, and Dr. Atienza's Deep Learning Experiments.
