from detector import encode_faces, recognise_face, validate
import argparse

# Create command line interface to interact with face_model

parser = argparse.ArgumentParser(
    description="Face Recognition Model"
)  # create parser object

parser.add_argument(
    "--train", dest="tr", action="store_true", help="Train the model"
)  # train
parser.add_argument(
    "--test",
    dest="t",
    action="store_true",
    help="Give location with -l along with --test to test the model, else use all images in validation",
)
parser.add_argument(
    "-l", "--location", action="store", help="Provide image location to test model"
)
parser.add_argument(
    "-m",
    "--model",
    action="store",
    default="hog",
    choices=["hog", "cnn"],
    help="Choose model to use",
)

args = parser.parse_args()  # collect

if args.tr:
    encode_faces(model=args.model)
if args.t and args.location:
    recognise_face(model=args.model, image_location=args.location)
elif args.t and not args.location:
    validate(model=args.model)
