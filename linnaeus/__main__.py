import argparse
import pathlib
from linnaeus.train import train
import linnaeus.linnaeus_ultima as linnaeus_ultima
from linnaeus.cli import predict
from copy import copy

parser = argparse.ArgumentParser(
    prog="Linnaeus CLI",
    description="This tool provides an easy interface for running linneaus ultima.",
    epilog="<Insert famous quote here>"
)

sub_parsers = parser.add_subparsers(dest="command")

ultima_parser = sub_parsers.add_parser("ultima", help="Run Linnaeus Ultima.")
ultima_parser.add_argument("image", help="The image to segment.")
ultima_parser.add_argument("--sam_checkpoint", default=linnaeus_ultima.DEFAULT_SAM_CHECKPOINT, help="The checkpoint to use for the SAM model.")
ultima_parser.add_argument("--model_type", default=linnaeus_ultima.DEFAULT_MODEL_TYPE, help="The type of model to use for SAM.")
ultima_parser.add_argument("--yolo_model", default=linnaeus_ultima.DEFAULT_FCOS_MODEL, help="The YOLO model to use.")
ultima_parser.add_argument("--device", default="cpu", help="The device to run the model on.")

train_parser = sub_parsers.add_parser("train")
train_parser.add_argument('weights', nargs="?", type=pathlib.Path)
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

command = copy(args.command)
del args.command

if command == "train":
    train(**vars(args))
elif command == "predict":
    predict.main(**vars(args))
elif command == "ultima":
    linnaeus_ultima.LinnaeusUltima.main(**vars(args))
else:
    exit(9)
