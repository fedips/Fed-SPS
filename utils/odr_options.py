
import argparse

def args_parser():
    parser = argparse.ArgumentParser()


    # epoch setting
    parser.add_argument('--global_epochs', type=int, default=10, help="rounds of training")
    parser.add_argument('--eval_epoch', type=int, default=50, help="rounds of evaluation")
    parser.add_argument('--save_epoch', type=int, default=50, help="rounds of saving models")
    parser.add_argument('--local_epochs', type=int, default=10, help="rounds of training")

    # federated setting
    parser.add_argument('--num_users', type=int, default=10, help="number of users: n")
    parser.add_argument('--shard_per_user', type=int, default=2, help="classes per user")
    parser.add_argument('--frac', type=float, default=0.1, help="the fraction of clients: C")
    parser.add_argument('--local_ep', type=int, default=5, help="the number of local epochs: E")
    parser.add_argument('--local_bs', type=int, default=10, help="local batch size: B")
    parser.add_argument('--bs', type=int, default=128, help="test batch size")
    parser.add_argument('--split', type=str, default='user', help="train-test split type, user or sample")
    parser.add_argument('--grad_norm', action='store_true', help='use_gradnorm_avging')
    parser.add_argument('--lr_decay', type=float, default=1.0, help="learning rate decay per round")
    parser.add_argument('--local_updates', type=int, default=1000000, help="maximum number of local updates")
    parser.add_argument('--m_tr', type=int, default=500, help="maximum number of samples/user to use for training")
    parser.add_argument('--m_ft', type=int, default=500, help="maximum number of samples/user to use for fine-tuning")

    # model arguments
    parser.add_argument('--model', type=str, default='mlp', help='model name')
    parser.add_argument('--kernel_num', type=int, default=9, help='number of each kind of kernel')
    parser.add_argument('--kernel_sizes', type=str, default='3,4,5',
                        help='comma-separated kernel size to use for convolution')
    parser.add_argument('--norm', type=str, default='batch_norm', help="batch_norm, layer_norm, or None")
    parser.add_argument('--num_filters', type=int, default=32, help="number of filters for conv nets")
    parser.add_argument('--max_pool', type=str, default='True',
                        help="Whether use max pooling rather than strided convolutions")
    parser.add_argument('--num_layers_keep', type=int, default=1, help='number layers to keep')
    parser.add_argument('--dropout', type=float, default = 0.0, help='whether dropout in Bi-LSTM')
    parser.add_argument('--hidden_dim', type=int, default=40, help='hidden dim in Bi-LSTM')
    parser.add_argument('--alg', type=str, default='fedavg', help='FL algorithm to use')

    # training arguments
    parser.add_argument('--perturb_w', action='store_true', help='whether perturbing the importance w or the distribution')
    parser.add_argument('--train_mode', type=str, default='', help='train mode local, global, odr')
    parser.add_argument('--mix_up', action='store_true', help='mix up')
    parser.add_argument('--GRL', type=float, default = 0.0, help='the lambda of the parameter')

    # odr hyperparameters for step size
    parser.add_argument('--lambda_odr', type=float, default='0.5', help='ODR parameter lambda')
    parser.add_argument('--alpha_odr', type=float, default='0.5', help='ODR parameter lambda')
    parser.add_argument('--beta_odr', type=float, default='0.5', help='ODR parameter lambda')
    parser.add_argument('--lr_odr_w', type=float, default='0.0001', help='ODR step size for updating w')
    parser.add_argument('--momentum', type=float, default=0.5, help="SGD momentum (default: 0.5)")
    parser.add_argument('--lr_global', type=float, default='0.1', help='learning rate during global training')
    parser.add_argument('--lr_local', type=float, default='0.1', help='learning rate during local training')

    # other arguments
    parser.add_argument('--dataset', type=str, default='mnist', help="name of dataset")
    parser.add_argument('--iid', action='store_true', help='whether i.i.d or not')
    parser.add_argument('--num_classes', type=int, default=10, help="number of classes")
    parser.add_argument('--num_channels', type=int, default=3, help="number of channels of imges")
    parser.add_argument('--gpu', type=int, default=0, help="GPU ID, -1 for CPU")
    parser.add_argument('--seed', type=int, default=1, help='random seed (default: 1)')
    parser.add_argument('--target_user', type=int, default = 0, help='random seed (default: 1)')

    # dir setting
    parser.add_argument('--dataset_dir', type=str, default='', help="dir of the data")
    parser.add_argument('--root_dir', type=str, default='', help="name for saving all files")
    parser.add_argument('--target_dir', type=str, default='', help="name for saving all files")
    parser.add_argument('--global_model_dir', type=str, default='global', help="dir for saving the global model")
    parser.add_argument('--local_model_dir', type=str, default='local', help="dir for saving the local model")

    # min-max setting
    parser.add_argument('--eps', type=float, default='0.03', help='eps for the distribution ball')
    parser.add_argument('--attack_steps', type=int, default=5, help="number of steps for generating adv samples.")
    parser.add_argument('--odr', action='store_true', help='whether use distributional robustness as regularizer')
    parser.add_argument('--select', action='store_true', help='whether select the more adversarial samples')

    # deploy
    parser.add_argument('--host_name', type=str, default='dog', help="name of the host")
    parser.add_argument('--auto_deploy', action='store_true', help='auto deploy all experiments')
    parser.add_argument('--log_dir', type=str, default='', help="name for saving all files")
  
    args = parser.parse_known_args()[0]
    return args
