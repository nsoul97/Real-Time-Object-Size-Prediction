# Real-Time-Object-Size-Prediction

#### How to train a new model?
```
usage: train.py [-h] --input {frames_number,movement_completion_perc} --strategy {all-in,one-out} --dataset_path DATASET_PATH [--seed SEED] --pretrain_task {forecast,value_imputation,mixed,no_pretrain}
                [--pretrain_masked_features {dependent,independent}] [--value_imp_mask_length VALUE_IMP_MASK_LENGTH] [--value_imp_mask_ratio VALUE_IMP_MASK_RATIO] [--forecast_mask_length FORECAST_MASK_LENGTH]
                [--std_dev_window STD_DEV_WINDOW] [--max_pad_length MAX_PAD_LENGTH] [--num_workers NUM_WORKERS] [--batch_size BATCH_SIZE] [--train_lr TRAIN_LR] [--train_epochs TRAIN_EPOCHS] [--pretrain_lr PRETRAIN_LR]
                [--pretrain_epochs PRETRAIN_EPOCHS] [--tsf_pos_encoder {learned,sinusoidal}] [--tsf_inp_encoder {conv_1d,linear}] [--tsf_norm {batch_norm,layer_norm}] [--tsf_blocks TSF_BLOCKS] [--tsf_heads TSF_HEADS]
                [--tsf_d_model TSF_D_MODEL] [--tsf_d_ff TSF_D_FF] [--tsf_dropout TSF_DROPOUT] [--tsf_act {relu,gelu}] [--tsf_conv_inp_encoder_kernel TSF_CONV_INP_ENCODER_KERNEL]

optional arguments:
  -h, --help            show this help message and exit
  --input {frames_number,movement_completion_perc}
                        The network either observes a fixed number of frames before making a prediction (real-time) or it observes a movement until a certain completion percentage (the end timeof the movement is known).
  --strategy {all-in,one-out}
                        The dataset split strategy to be used.
  --dataset_path DATASET_PATH
                        The path of the movement dataset directory 'object_size_prediction/dataset/'.
  --seed SEED           A random seed to reproduce the results. (default: 0)
  --pretrain_task {forecast,value_imputation,mixed,no_pretrain}
                        The task that will be used for pretraining the network. To skip pretraining, select the 'no_pretrain' option.
  --pretrain_masked_features {dependent,independent}
                        The features that are masked in pre-training can be either dependent or independent to each other. If the features are dependent, we select the frames that will be masked once and all the features are
                        masked for these frames. If the features are independent, we select the frames that will be masked for each feature independently. (default: independent)
  --value_imp_mask_length VALUE_IMP_MASK_LENGTH
                        The average length of the masked segments for the value imputation and the mixed pretraining task. (default: 3)
  --value_imp_mask_ratio VALUE_IMP_MASK_RATIO
                        The ratio r = (# masked frames) / (# movement frames) of the masked frames for the value imputation and the mixed pretraining task. (default: 0.25)
  --forecast_mask_length FORECAST_MASK_LENGTH
                        The average length of the masked frames' segment at the end of the movement for the forecasting and the mixed pretraining task. (default: 15)
  --std_dev_window STD_DEV_WINDOW
                        The wrist-y standard deviation for a given frame is calculated using a sliding window. The wrist-y coordinates of the current frame and of the (window-1) previous frames are used. (default: 10)
  --max_pad_length MAX_PAD_LENGTH
                        The number of frames T after padding. If a movement has M frames, the T-M frames on the right are padded with zeros and each movement can be represented as a (T, F)-tensor, where F is the number of
                        features that are extracted for a frame. T must be > M for all the movements. (default: 70)
  --num_workers NUM_WORKERS
                        The number of workers used to load the data from the training and the test set. (default: 2)
  --batch_size BATCH_SIZE
                        The number of movements that are loaded by the dataloaders in a single mini-batch. (default: 32)
  --train_lr TRAIN_LR   The learning rate used for training (classification task). (default: 0.0005)
  --train_epochs TRAIN_EPOCHS
                        The number of epochs the model is trained for (classification task). (default: 100)
  --pretrain_lr PRETRAIN_LR
                        The learning rate used for pretraining (value imputation, forecasting or mixed task). (to be specified only when the network is pretrained). (default: 0.0005)
  --pretrain_epochs PRETRAIN_EPOCHS
                        The number of epochs the model is pretrained for (value imputation, forecasting or mixed task) (to be specified only when the network is pretrained). (default: 5000)
  --tsf_pos_encoder {learned,sinusoidal}
                        The positional encoder can either be learned or be sinusoidal. (default: learned)
  --tsf_inp_encoder {conv_1d,linear}
                        The input's frames are encoded using either a 1D convolution, which considers the temporally adjacent frames, or a FC layer, which considers only the current frame. (default: conv_1d)
  --tsf_norm {batch_norm,layer_norm}
                        The output of the multi-head attention block and of the feed-forward block are normalized using either batch norm or layer norm. (default: layer_norm)
  --tsf_blocks TSF_BLOCKS
                        The number of blocks of the transformer encoder. (default: 3)
  --tsf_heads TSF_HEADS
                        The number of heads in the multi-head attention mechanism of each transformer block. (default: 16)
  --tsf_d_model TSF_D_MODEL
                        The model dimension D. Each (T, F) input tensor is encoded to a (T, D) tensor by the input encoder (analogous to the word embeddings in NLP transformers). (default: 128)
  --tsf_d_ff TSF_D_FF   The number of neurons in the first layer of the feed forward block. (default: 256)
  --tsf_dropout TSF_DROPOUT
                        The dropout probability. (default: 0.1)
  --tsf_act {relu,gelu}
                        The activation function that is used in the transformer encoded can be either ReLU or GELU. (default: relu)
  --tsf_conv_inp_encoder_kernel TSF_CONV_INP_ENCODER_KERNEL
                        The kernel size of 1D-convolutional input encoder. (to be specified only when tsf_inp_encoder = 'conv_1d'). (default: 5)
```

#### How to check the experiments' logs?
```
tensorboard --logdir logs
```
