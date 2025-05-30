from pathlib import Path
import dlib  # library for computer vision built in c++
import face_recognition  # library or wrapper built upono dlib
import pickle
from collections import Counter
from PIL import Image, ImageDraw, ImageFont


# CONSTANTS
DEFAULT_ENCODED_PATH = Path("output\\encoded.pkl")
RECT_OUTLINE_COLOR = "blue"
TEXT_COLOR = "white"


def encode_faces(
    model: str = "hog", encoding_location: Path = DEFAULT_ENCODED_PATH
) -> None:
    """Function to train model and encode data"""
    names = []
    encodings = []

    for filename in Path("training").glob("*/*"):
        name = filename.parent.name
        image = face_recognition.load_image_file(filename)
        face_location = face_recognition.face_locations(image, model=model)
        face_encoding = face_recognition.face_encodings(image, face_location)

        for encoding in face_encoding:
            names.append(name)
            encodings.append(encoding)

    name_encodings = {"names": names, "encodings": encodings}

    with encoding_location.open(mode="wb") as f:
        pickle.dump(name_encodings, f)


def recognise_face(
    image_location: str,
    model: str = "hog",
    encoding_location: Path = DEFAULT_ENCODED_PATH,
):
    """Method to recognise face, takes image location, model and sotred data location"""

    # loading saved data
    with encoding_location.open(mode="rb") as f:
        loaded_encodings = pickle.load(f)

    # getting location and encodings of new image
    input_image = face_recognition.load_image_file(
        Path(f"validation\\{image_location}")
    )
    input_image_locations = face_recognition.face_locations(input_image, model=model)
    input_image_encodings = face_recognition.face_encodings(
        input_image, input_image_locations
    )

    pillow_image = Image.fromarray(input_image)  # create pillow image to open
    draw = ImageDraw.Draw(
        pillow_image
    )  # create draw object to draw rect and text on image

    for bounding_box, unknown_encoding in zip(
        input_image_locations, input_image_encodings
    ):
        name = _recognise_face(unknown_encoding, loaded_encodings)
        if not name:
            name = "Unknown"
        _display_face(draw, bounding_box, name)

        del draw  # deleting draw object
        pillow_image.show()


def _display_face(draw, bounding_box, name, font_size=20):
    """Draws face bounding box and name with adjustable text size."""
    top, right, bottom, left = bounding_box

    # Draw face bounding box
    draw.rectangle(((left, top), (right, bottom)), outline=RECT_OUTLINE_COLOR)

    # Load a font with the desired size
    try:
        font = ImageFont.truetype("arial.ttf", font_size)  # Use arial if available
    except IOError:
        font = ImageFont.load_default()  # Fallback to default (small)

    # Get text bounding box (now respects font size)
    text_left, text_top, text_right, text_bottom = draw.textbbox(
        (left, bottom), name, font=font
    )

    # Draw background rectangle for text (larger due to bigger font)
    draw.rectangle(
        ((text_left, text_top), (text_right, text_bottom)),
        fill=RECT_OUTLINE_COLOR,
        outline=RECT_OUTLINE_COLOR,
    )

    # Draw the text (now larger)
    draw.text(
        (text_left, text_top),
        name,
        fill=TEXT_COLOR,
        font=font,  # Apply the larger font
    )


def _recognise_face(unknown_encoding, loaded_encodings):
    boolean_matches = face_recognition.compare_faces(
        loaded_encodings["encodings"], unknown_encoding
    )
    votes = Counter(
        name for match, name in zip(boolean_matches, loaded_encodings["names"]) if match
    )
    if votes:
        return votes.most_common(1)[0][0]


def validate(model: str = "hog"):
    path = Path("validation")
    for p in path.iterdir():
        recognise_face(p.name, model=model)
