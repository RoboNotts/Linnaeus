import argparse
import pathlib

parser = argparse.ArgumentParser(
    prog="FCOS Command Line Tool",
    description="This tool provides an easy interface for fcos training and running",
    epilog="<Insert famous quote here>"
)

sub_parsers = parser.add_subparsers(dest="command")

train_parser = sub_parsers.add_parser("train")
train_parser.add_argument('weights', type=pathlib.Path)
train_parser.add_argument('classfile', type=pathlib.Path)
train_parser.add_argument('train_dataset', type=pathlib.Path)
train_parser.add_argument('val_dataset', type=pathlib.Path)
train_parser.add_argument('--batch_size', default=2, type=int, required=False)
train_parser.add_argument('--epoch', type=int, default=1000, required=False)
train_parser.add_argument('--lr', type=float, default=0.0001, required=False)
train_parser.add_argument('--ft_lr', type=float, default=0.000001, required=False)
train_parser.add_argument('--start', type=int, default=0, required=False)
train_parser.add_argument('--weight_decay', type=float, default=0.005, required=False)
train_parser.add_argument('--optimizer_name', type=str, default="Adam",  required=False)
train_parser.add_argument('--save_file', type=pathlib.Path, required=False)

predict_parser = sub_parsers.add_parser("predict")
predict_parser.add_argument('model', type=pathlib.Path)
predict_parser.add_argument('weights', type=pathlib.Path)
predict_parser.add_argument('classfile', type=pathlib.Path)
predict_parser.add_argument('image', type=pathlib.Path)

args = parser.parse_args()

if args.command == "train":
    try:
        from fcos.train import train
        del args.command
        
        train(**vars(args))
    except ImportError:
        print("Train Module not included.")
elif args.command == "predict":
    from fcos.cli import predict
    del args.command

    predict.main(**vars(args))
