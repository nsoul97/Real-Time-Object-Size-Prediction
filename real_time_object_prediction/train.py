import os.path
import time
import numpy as np
import torch as th
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data_utils
import reproducibility
from data import Data, get_dataset_statistics, PretrainDataset, PartiallyObservableDataset
from torch.utils.tensorboard import SummaryWriter
from model import GraspMovNet
from typing import Literal, Optional
import argparse


def pretrain(pretrain_net: GraspMovNet,
             train_set: data_utils.Dataset,
             mean: th.Tensor,
             std_dev: th.Tensor,
             epochs: int,
             batch_size: int,
             num_workers: int,
             lr: float,
             device: Literal['cpu', 'cuda'],
             task_mode: Literal['forecast', 'value_imputation', 'mixed'],
             mask_mode: Literal['independent', 'dependent'],
             summary_writer: SummaryWriter,
             summary_description: str,
             value_imp_mask_length: Optional[int] = None,
             value_imp_ratio: Optional[float] = None,
             forecast_mask_length: Optional[int] = None,
             ) -> None:
    mean = mean.to(device)
    std_dev = std_dev.to(device)
    pretrain_net.to(device)

    assert pretrain_net.mode == 'pretrain', "The network must be set in 'pretrain' mode."
    train_set = PretrainDataset(train_set,
                                task_mode=task_mode,
                                mask_mode=mask_mode,
                                value_imp_mask_length=value_imp_mask_length,
                                value_imp_ratio=value_imp_ratio,
                                forecast_mask_length=forecast_mask_length)

    train_loader = data_utils.DataLoader(train_set,
                                         batch_size=batch_size,
                                         num_workers=num_workers,
                                         shuffle=True,
                                         worker_init_fn=reproducibility.seed_worker,
                                         generator=reproducibility.rng['th_rng'])

    criterion = nn.MSELoss(reduction='mean')
    optimizer = optim.Adam(pretrain_net.parameters(), lr=lr)

    pretrain_net.train()
    for e in range(epochs):
        epoch_train_loss = 0

        for x, pad_mask, value_mask in train_loader:
            x = x.to(device)
            pad_mask = pad_mask.to(device)
            value_mask = value_mask.to(device)

            x_norm = (x - mean) / std_dev
            x_norm = x_norm.masked_fill(value_mask == 0, 0)
            y = pretrain_net(x_norm, pad_mask)

            y_pred = y.masked_select(value_mask == 0)
            y_gt = x.masked_select(value_mask == 0)

            loss = criterion(y_pred, y_gt)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_train_loss += loss.item()

        epoch_train_loss /= len(train_loader)
        summary_writer.add_scalar(f'{summary_description}: Avg Epoch Loss', epoch_train_loss, e + 1)


def train_classification(model: GraspMovNet,
                         train_set: data_utils.Dataset,
                         test_set: data_utils.Dataset,
                         mean: th.Tensor,
                         std_dev: th.Tensor,
                         epochs: int,
                         batch_size: int,
                         num_workers: int,
                         lr: float,
                         device: Literal['cpu', 'cuda'],
                         summary_writer: SummaryWriter,
                         summary_description: str,
                         keep_frames: Optional[int] = None,
                         mov_completion_perc: Optional[float] = None) -> float:
    train_set = PartiallyObservableDataset(train_set,
                                           keep_frames=keep_frames,
                                           mov_completion_perc=mov_completion_perc)

    test_set = PartiallyObservableDataset(test_set,
                                          keep_frames=keep_frames,
                                          mov_completion_perc=mov_completion_perc)

    mean = mean.to(device)
    std_dev = std_dev.to(device)
    model.to(device)

    assert model.mode == 'train', "The network must be set in 'train' mode."

    train_loader = data_utils.DataLoader(train_set,
                                         batch_size=1, # batch_size,
                                         num_workers=num_workers,
                                         shuffle=True,
                                         worker_init_fn=reproducibility.seed_worker,
                                         generator=reproducibility.rng['th_rng'])

    test_loader = data_utils.DataLoader(test_set,
                                        batch_size=batch_size,
                                        num_workers=num_workers,
                                        worker_init_fn=reproducibility.seed_worker,
                                        generator=reproducibility.rng['th_rng'])

    criterion = nn.CrossEntropyLoss(reduction='mean')
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for e in range(epochs):

        epoch_train_loss = 0
        epoch_train_avg_acc = 0
        model.train()
        for x, pad_mask, y_gt in train_loader:
            x = x.to(device)
            pad_mask = pad_mask.to(device)
            y_gt = y_gt.to(device)

            x_norm = (x - mean) / std_dev
            x_norm = x_norm.masked_fill(pad_mask[..., None] == 0, 0)

            y_pred = model(x_norm)
            loss = criterion(y_pred, y_gt)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_train_avg_acc += th.sum((th.argmax(y_pred, dim=1) == y_gt).float()).item()
            epoch_train_loss += loss.item()

        epoch_train_loss /= len(train_loader)
        epoch_train_avg_acc /= len(train_set)

        epoch_test_loss = 0
        epoch_test_avg_acc = 0
        model.eval()
        with th.no_grad():
            for x, pad_mask, y_gt in test_loader:
                x = x.to(device)
                pad_mask = pad_mask.to(device)
                y_gt = y_gt.to(device)

                x_norm = (x - mean) / std_dev
                x_norm = x_norm.masked_fill(pad_mask[..., None] == 0, 0)

                y_pred = model(x_norm, pad_mask)
                loss = criterion(y_pred, y_gt)

                epoch_test_avg_acc += th.sum((th.argmax(y_pred, dim=1) == y_gt).float()).item()
                epoch_test_loss += loss.item()

        epoch_test_loss /= len(test_loader)
        epoch_test_avg_acc /= len(test_set)

        summary_writer.add_scalar(f'{summary_description}: Train set: Avg Epoch Loss', epoch_train_loss, e + 1)
        summary_writer.add_scalar(f'{summary_description}: Test set: Avg Epoch Loss', epoch_test_loss, e + 1)
        summary_writer.add_scalar(f'{summary_description}: Train set: Accuracy', epoch_train_avg_acc * 100, e + 1)
        summary_writer.add_scalar(f'{summary_description}: Test set: Accuracy', epoch_test_avg_acc * 100, e + 1)

    return epoch_test_avg_acc


def parse_args():
    DEF_INPUT_CONV_F = 5
    DEF_PRETRAIN_LR = 5e-4
    DEF_PRETRAIN_EPOCHS = 5000
    DEF_PRETRAIN_MASKED_FEATURES = 'independent'
    DEF_VAL_IMP_MASK_LENGTH = 3
    DEF_VAL_IMP_RATIO = 0.25
    DEF_FORECAST_MASK_LENGTH = 15

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--input', type=str, choices=['frames_number', 'movement_completion_perc'],
                        default=argparse.SUPPRESS,
                        help='The network either observes a fixed number of frames before making a prediction '
                             '(real-time) or it observes a movement until a certain completion percentage (the end time'
                             'of the movement is known).',
                        required=True)

    parser.add_argument('--strategy', type=str, choices=['all-in', 'one-out'], default=argparse.SUPPRESS,
                        help='The dataset split strategy to be used.',
                        required=True)

    parser.add_argument('--dataset_path', type=str, default=argparse.SUPPRESS,
                        help="The path of the movement dataset directory 'object_size_prediction/dataset/'.",
                        required=True)

    parser.add_argument("--seed", type=int, default=0,
                        help="A random seed to reproduce the results.")

    parser.add_argument("--pretrain_task", type=str, choices=['forecast', 'value_imputation', 'mixed', 'no_pretrain'],
                        required=True, default=argparse.SUPPRESS,
                        help="The task that will be used for pretraining the network. To skip pretraining, select the "
                             "'no_pretrain' option.")

    parser.add_argument("--pretrain_masked_features", type=str, choices=['dependent', 'independent'],
                        default=argparse.SUPPRESS,
                        help="The features that are masked in pre-training can be either dependent or independent to "
                             "each other. If the features are dependent, we select the frames that will be masked once "
                             "and all the features are masked for these frames. If the features are independent, we "
                             "select the frames that will be masked for each feature independently. (default: "
                             f"{DEF_PRETRAIN_MASKED_FEATURES})")

    parser.add_argument("--value_imp_mask_length", type=int, default=argparse.SUPPRESS,
                        help="The average length of the masked segments for the value imputation and the mixed "
                             f"pretraining task. (default: {DEF_VAL_IMP_MASK_LENGTH})")

    parser.add_argument("--value_imp_mask_ratio", type=float, default=argparse.SUPPRESS,
                        help="The ratio r = (# masked frames) / (# movement frames) of the masked frames for the value "
                             f"imputation and the mixed pretraining task. (default: {DEF_VAL_IMP_RATIO})")

    parser.add_argument("--forecast_mask_length", type=int, default=argparse.SUPPRESS,
                        help="The average length of the masked frames' segment at the end of the movement for the "
                             f"forecasting and the mixed pretraining task. (default: {DEF_FORECAST_MASK_LENGTH})")

    parser.add_argument('--std_dev_window', type=int, default=10,
                        help='The wrist-y standard deviation for a given frame is calculated using a sliding window. '
                             'The wrist-y coordinates of the current frame and of the (window-1) previous frames are '
                             'used.')

    parser.add_argument('--max_pad_length', type=int, default=70,
                        help='The number of frames T after padding. If a movement has M frames, the T-M frames on the '
                             'right are padded with zeros and each movement can be represented as a (T, F)-tensor, '
                             'where F is the number of features that are extracted for a frame. T must be > M for all '
                             'the movements.')

    parser.add_argument('--num_workers', type=int, default=2,
                        help='The number of workers used to load the data from the training and the test set.')

    parser.add_argument('--batch_size', type=int, default=32,
                        help='The number of movements that are loaded by the dataloaders in a single mini-batch.')

    parser.add_argument('--train_lr', type=float, default=5e-4,
                        help='The learning rate used for training (classification task).')

    parser.add_argument('--train_epochs', type=int, default=100,
                        help='The number of epochs the model is trained for (classification task).')

    parser.add_argument('--pretrain_lr', type=float, default=argparse.SUPPRESS,
                        help='The learning rate used for pretraining (value imputation, forecasting or mixed task). '
                             f'(to be specified only when the network is pretrained). (default: {DEF_PRETRAIN_LR})')

    parser.add_argument('--pretrain_epochs', type=int, default=argparse.SUPPRESS,
                        help='The number of epochs the model is pretrained for (value imputation, forecasting or'
                             f' mixed task) (to be specified only when the network is pretrained). (default:'
                             f' {DEF_PRETRAIN_EPOCHS})')

    parser.add_argument('--tsf_pos_encoder', type=str, choices=['learned', 'sinusoidal'], default='learned',
                        help='The positional encoder can either be learned or be sinusoidal.')

    parser.add_argument('--tsf_inp_encoder', type=str, choices=['conv_1d', 'linear'], default='conv_1d',
                        help="The input's frames are encoded using either a 1D convolution, which considers the "
                             "temporally adjacent frames, or a FC layer, which considers only the current frame.")

    parser.add_argument('--tsf_norm', type=str, choices=['batch_norm', 'layer_norm'], default='layer_norm',
                        help='The output of the multi-head attention block and of the feed-forward block are normalized'
                             ' using either batch norm or layer norm.')

    parser.add_argument('--tsf_blocks', type=int, default=3,
                        help='The number of blocks of the transformer encoder.')

    parser.add_argument('--tsf_heads', type=int, default=16,
                        help='The number of heads in the multi-head attention mechanism of each transformer block.')

    parser.add_argument('--tsf_d_model', type=int, default=128,
                        help='The model dimension D. Each (T, F) input tensor is encoded to a (T, D) tensor by the '
                             'input encoder (analogous to the word embeddings in NLP transformers).')

    parser.add_argument('--tsf_d_ff', type=int, default=256,
                        help='The number of neurons in the first layer of the feed forward block.')

    parser.add_argument('--tsf_dropout', type=float, default=0.1,
                        help='The dropout probability.')

    parser.add_argument('--tsf_act', type=str, choices=['relu', 'gelu'], default='relu',
                        help='The activation function that is used in the transformer encoded can be either ReLU or '
                             'GELU.')

    parser.add_argument('--tsf_conv_inp_encoder_kernel', type=int, default=argparse.SUPPRESS,
                        help="The kernel size of 1D-convolutional input encoder. (to be specified only when "
                             f"tsf_inp_encoder = 'conv_1d'). (default: {DEF_INPUT_CONV_F})")

    params = vars(parser.parse_args())

    if params['pretrain_task'] in ['no_pretrain', 'value_imputation']:
        assert 'forecast_mask_length' not in params, "'forecast_mask_length' can be specified only when pretraining " \
                                                     "the network for the forecasting or the mixed task."
        params['forecast_mask_length'] = None
    else:
        if 'forecast_mask_length' not in params: params['forecast_mask_length'] = DEF_FORECAST_MASK_LENGTH

    if params['pretrain_task'] in ['no_pretrain', 'forecast']:
        assert 'value_imp_mask_length' not in params, "'value_imp_mask_length' can be specified only when pretraining" \
                                                      " the network for the value imputation or the mixed task."
        assert 'value_imp_mask_ratio' not in params, "'value_imp_mask_ratio' can be specified only when pretraining" \
                                                     " the network for the value imputation or the mixed task."
        params['value_imp_mask_length'] = None
        params['value_imp_mask_ratio'] = None
    else:
        if 'value_imp_mask_length' not in params: params['value_imp_mask_length'] = DEF_VAL_IMP_MASK_LENGTH
        if 'value_imp_mask_ratio' not in params: params['value_imp_mask_ratio'] = DEF_VAL_IMP_RATIO

    if params['pretrain_task'] == 'no_pretrain':
        assert 'pretrain_lr' not in params, "The network will not be pretrained. 'pretrain_lr' must not be specified."
        assert 'pretrain_epochs' not in params, "The network will not be pretrained. 'pretrain_epochs' must not be " \
                                                "specified."
        assert 'pretrain_masked_features' not in params, "The network will not be pretrained. " \
                                                         "'pretrain_masked_features' must not be specified."
        params['pretrain_lr'] = None
        params['pretrain_epochs'] = None
        params['pretrain_masked_features'] = None
    else:
        if 'pretrain_lr' not in params: params['pretrain_lr'] = DEF_PRETRAIN_LR
        if 'pretrain_epochs' not in params: params['pretrain_epochs'] = DEF_PRETRAIN_EPOCHS
        if 'pretrain_masked_features' not in params: params['pretrain_masked_features'] = DEF_PRETRAIN_MASKED_FEATURES

    if params['tsf_inp_encoder'] == 'conv_1d':
        if 'tsf_conv_inp_encoder_kernel' not in params:
            params['tsf_conv_inp_encoder_kernel'] = DEF_INPUT_CONV_F
    else:
        assert 'tsf_conv_inp_encoder_kernel' not in params, "The input encoder is not convolutional. " \
                                                            "'tsf_conv_inp_encoder_kernel' must not be specified."
        params['tsf_conv_inp_encoder_kernel'] = None

    return params


def main():
    params = parse_args()

    log_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../logs/'))
    if os.path.isdir(log_path) is False: os.mkdir(log_path)
    cur_log_subdir = os.path.join(log_path, time.strftime('%d_%m_%Y__%H_%M_%S'))
    writer = SummaryWriter(log_dir=cur_log_subdir)

    device = 'cuda' if th.cuda.is_available() else 'cpu'
    reproducibility.init_rng(params['seed'])

    data = Data(dataset_path=params['dataset_path'],
                strategy=params['strategy'],
                max_pad_length=params['max_pad_length'],
                std_dev_window=params['std_dev_window'],
                d_input=4)

    accuracies = dict()
    for p, (train_set, test_set) in enumerate(data):
        mean, std_dev = get_dataset_statistics(train_set)
        model = GraspMovNet(N=params['tsf_blocks'],
                            heads=params['tsf_heads'],
                            d_input=4,
                            d_output=3,
                            max_pad_length=params['max_pad_length'],
                            d_model=params['tsf_d_model'],
                            d_ff=params['tsf_d_ff'],
                            dropout=params['tsf_dropout'],
                            kernel_size=params['tsf_conv_inp_encoder_kernel'],
                            activation=params['tsf_act'],
                            mode='pretrain',
                            pos_encoder_mode=params['tsf_pos_encoder'],
                            norm_mode=params['tsf_norm'],
                            input_enc_mode=params['tsf_inp_encoder'])

        if params['pretrain_task'] != 'no_pretrain':
            pretrain(pretrain_net=model,
                     train_set=train_set,
                     mean=mean,
                     std_dev=std_dev,
                     epochs=params['pretrain_epochs'],
                     batch_size=params['batch_size'],
                     num_workers=params['num_workers'],
                     lr=params['pretrain_lr'],
                     device=device,
                     task_mode=params['pretrain_task'],
                     mask_mode=params['pretrain_masked_features'],
                     value_imp_mask_length=params['value_imp_mask_length'],
                     value_imp_ratio=params['value_imp_mask_ratio'],
                     forecast_mask_length=params['forecast_mask_length'],
                     summary_writer=writer,
                     summary_description=f'Pretraining: '
                                         f'k-fold partition {p + 1}')
        model.set_mode('train')

        if params['input'] == 'movement_completion_perc':
            for mov_completion_perc in [0.2, 0.4, 0.6, 0.8, 1.0]:
                acc = train_classification(model=model,
                                           train_set=train_set,
                                           test_set=test_set,
                                           mean=mean,
                                           std_dev=std_dev,
                                           epochs=params['train_epochs'],
                                           batch_size=params['batch_size'],
                                           num_workers=params['num_workers'],
                                           lr=params['train_lr'],
                                           device=device,
                                           mov_completion_perc=mov_completion_perc,
                                           summary_writer=writer,
                                           summary_description=f'Training: '
                                                               f'k-fold partition {p + 1}: '
                                                               f'Movement Completion Percentage {mov_completion_perc * 100:.0f}%')

                print(f'Movement Completion Percentage: {mov_completion_perc * 100:.0f}%, '
                      f'k-fold partition: {p + 1}, '
                      f'accuracy: {acc}')

                if mov_completion_perc in accuracies:
                    accuracies[mov_completion_perc].append(acc)
                else:
                    accuracies[mov_completion_perc] = [acc]

        else:
            for keep_frames in [10, 15, 20, 25]:
                acc = train_classification(model=model,
                                           train_set=train_set,
                                           test_set=test_set,
                                           mean=mean,
                                           std_dev=std_dev,
                                           epochs=params['train_epochs'],
                                           batch_size=params['batch_size'],
                                           num_workers=params['num_workers'],
                                           lr=params['train_lr'],
                                           device=device,
                                           keep_frames=keep_frames,
                                           summary_writer=writer,
                                           summary_description=f'Training: '
                                                               f'k-fold partition {p + 1}: '
                                                               f'Keep Frames {keep_frames}')

                print(f'Keep Frames: {keep_frames}, '
                      f'k-fold partition: {p + 1}, '
                      f'accuracy: {acc}')

                if keep_frames in accuracies:
                    accuracies[keep_frames].append(acc)
                else:
                    accuracies[keep_frames] = [acc]

    metrics = dict()
    if params['input'] == 'movement_completion_perc':
        for mov_completion_perc in [0.2, 0.4, 0.6, 0.8, 1.0]:
            mean_acc = np.mean(accuracies[mov_completion_perc])
            stddev_acc = np.std(accuracies[mov_completion_perc])

            metrics[f'avg accuracy {mov_completion_perc * 100:.0f}'] = mean_acc.item()
            metrics[f'stddev_accuracy {mov_completion_perc * 100:.0f}'] = stddev_acc.item()

            print('====================================================================')
            print(f'Movement Completion Percentage: {mov_completion_perc * 100:.0f}%')
            print('Accuracy/k-fold partition: ', accuracies[mov_completion_perc])
            print('Mean Accuracy: ', mean_acc)
            print('Accuracy Standard Deviation: ', stddev_acc)
    else:
        for keep_frames in [10, 15, 20, 25]:
            mean_acc = np.mean(accuracies[keep_frames])
            stddev_acc = np.std(accuracies[keep_frames])

            metrics[f'avg accuracy {keep_frames}'] = mean_acc.item()
            metrics[f'stddev_accuracy {keep_frames}'] = stddev_acc.item()

            print('====================================================================')
            print(f'Keep Frames: {keep_frames}')
            print('Accuracy/k-fold partition: ', accuracies[keep_frames])
            print('Mean Accuracy: ', mean_acc)
            print('Accuracy Standard Deviation: ', stddev_acc)

    writer.add_hparams(hparam_dict=params, metric_dict=metrics)


if __name__ == '__main__':
    main()
