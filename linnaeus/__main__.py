import argparse
import linnaeus.linnaeus_ultima as linnaeus_ultima

parser = argparse.ArgumentParser(
    prog="Linnaeus CLI",
    description="This tool provides an easy interface for running linneaus ultima.",
    epilog="<Insert famous quote here>"
)

parser.add_argument("image", help="The image to segment.")
parser.add_argument("--sam_checkpoint", default=linnaeus_ultima.DEFAULT_SAM_CHECKPOINT, help="The checkpoint to use for the SAM model.")
parser.add_argument("--model_type", default=linnaeus_ultima.DEFAULT_MODEL_TYPE, help="The type of model to use for SAM.")
parser.add_argument("--yolo_model", default=linnaeus_ultima.DEFAULT_YOLO_MODEL, help="The YOLO model to use.")
parser.add_argument("--device", default="cpu", help="The device to run the model on.")

args = parser.parse_args()

linnaeus_ultima.LinnaeusUltima.main(**vars(args))
