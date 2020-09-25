import argparse
from model import Model

parser = argparse.ArgumentParser(description="CondConv args")
parser.add_argument('--dataset',
                    type=str,
                    help='name of  dataset',
                    default="cifar10",
                    choices=["cifar10", "cifar100"]
                    )
parser.add_argument('--data_path',
                    type=str,
                    help='path to dataset',
                    default=""
                    )
parser.add_argument("--lr",
                    type=float,
                    default=0.1
                    )
parser.add_argument("--train_batch",
                    type=int,
                    default=128
                    )
parser.add_argument("--val_batch",
                    type=int,
                    default=128
                    )
parser.add_argument("--epochs",
                    type=int,
                    default=200
                    )
parser.add_argument("--arch",
                    type=str,
                    default="cond_cifar_resnet",
                    choices=["cond_cifar_resnet"]
                    )
parser.add_argument("--num_layers",
                    type=int,
                    default=56,
                    choices=[20, 32, 44, 56, 110, 1202]
                    )
parser.add_argument("--num_experts",
                    type=int,
                    default=3
                    )
parser.add_argument("--num_classes",
                    type=int,
                    default=10
                    )
parser.add_argument("--models_path",
                    type=str,
                    default="./models"
                    )
parser.add_argument("--logs_path",
                    type=str,
                    default="./logs"
                    )
# list of available gpu
parser.add_argument("--gpu_ids",
                    nargs='+',
                    default=['0']
                    )
parser.add_argument("--val",
                    action="store_true"
                    )
parser.add_argument("--pretrained",
                    action="store_true"
                    )
parser.add_argument("--distribute",
                    help='train model in distributed way',
                    action="store_true"
                    )
parser.add_argument("--resume",
                    help='whether to load the latest checkpoint',
                    action="store_true"
                    )
args = parser.parse_args()

if __name__ == '__main__':
    model = Model(args)
    model.main()
