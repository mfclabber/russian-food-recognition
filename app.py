import streamlit as st
from PIL import Image
from pathlib import Path
import numpy as np
import cv2
import random
import torch
import torchvision.transforms as T
import os, urllib
import matplotlib.pyplot as plt

from scripts.utils import *
from scripts.model import SSD_MobileNetV3

IMAGE_SHAPE = (640, 480)

APP_NAME = "AlfaFood SSD_MobileNetV3.py"

MODEL_WEIGHTS_URL = (
    ""
)

MODEL_PATH = "weights/best_model.pth"

DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    
SAMPLE_IMAGES = [
    Path("sample_images/0.jpg"),
    Path("sample_images/1.jpg"),
    Path("sample_images/2.jpg"),
    Path("sample_images/3.jpg"),
    Path("sample_images/4.jpg"),
]

NUM_CLASSES = 128


COLORS = list((random.randint(40, 240), random.randint(40, 255), random.randint(60, 255)) for i in range(129))


@st.cache_data
def get_file_content_as_string(path):
    """
    Download a single file and make its content available as a string.
    """
    with open(path, encoding="utf-8", errors="ignore") as f:
        response = f.read()
    return response


def show_image(image_path):
    """ Show an image """
    image = load_image_file(image_path)
    if image_path[-3:] == "jpg":
        st.image(image, caption="", use_column_width=True, clamp=True, format="JPEG")
    elif image_path[-3:] == "png":
        st.image(image, caption="", use_column_width=True, clamp=True, format="PNG")
    elif image_path[-4:] == "jpeg":
        st.image(image, caption="", use_column_width=True, clamp=True, format="JPEG")
    else:
        print("Invalid Image")


@st.cache_data
def load_image_url(url):
    """ Loads an image given the url """
    with urllib.request.urlopen(url) as response:
        image = np.asarray(bytearray(response.read()), dtype="uint8")
    image = cv2.imdecode(image, cv2.IMREAD_COLOR)
    image = image[:, :, [2, 1, 0]]  # BGR -> RGB

    transform = T.Compose(
                          T.Resize(480, 640),
                          T.ToTensor(image)
                         )
    image_tensor = transform(image).to(DEVICE)
    return image_tensor


@st.cache_data
def load_image_tensor(image_path, device):
    """
    Loads an image into pytorch tensor. 
    """
    image_tensor = T.ToTensor()(Image.open(image_path)).to(device)
    return image_tensor 


@st.cache_data
def load_image_file(image_path):
    """
    Loads an Image file
    """
    image = Image.open(image_path)
    image = image.convert("RGB")
    image = np.array(image)
    image = image / 255.0
    image = image.astype(np.float32)
    return image


def download_file(file_path, save_path):
    """
    Utility to beautifully download a file from its url
    """
    # Don't download the file twice. (If possible, verify the download using the file length.)
    if os.path.exists(save_path):
        return
    else:
        weights_warning, progress_bar = None, None
        try:
            weights_warning = st.warning("Downloading %s..." % file_path)
            progress_bar = st.progress(0)
            with open(save_path, "wb") as output_file:
                with urllib.request.urlopen(file_path) as response:
                    length = int(response.info()["Content-Length"])
                    counter = 0.0
                    MEGABYTES = 2.0 ** 20.0
                    while True:
                        data = response.read(8192)
                        if not data:
                            break
                        counter += len(data)
                        output_file.write(data)

                        weights_warning.warning(
                            "Downloading %s... (%6.2f/%6.2f MB)"
                            % (file_path, counter / MEGABYTES, length / MEGABYTES)
                        )
                        progress_bar.progress(min(counter / length, 1.0))
        finally:
            if weights_warning is not None:
                weights_warning.empty()
            if progress_bar is not None:
                progress_bar.empty()


def create_model():
    """Initialize model"""
    model = SSD_MobileNetV3(num_classes=NUM_CLASSES)
    return model


@st.cache_data
def load_model(model_path):
    """ Create the model and load state dict here """
    model = create_model()
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.eval()  # Set to eval mode
    model.to(DEVICE)
    return model


# TODO

def objects_threshold_scores(bboxes: torch.Tensor, 
                         labels: torch.Tensor=None, 
                         scores: torch.Tensor=None,
                         threshold_score: float=0.1):
    bboxes_copy = copy.deepcopy(bboxes)
    labels_copy = copy.deepcopy(labels)
    scores_copy = copy.deepcopy(scores)

    bboxes = torch.Tensor([])
    labels, scores = list(), list()
    for i, score in enumerate(scores_copy):
        if score >= threshold_score:
            bboxes = torch.cat((bboxes, bboxes_copy[i].unsqueeze(dim=0)), dim=0)
            labels.append(labels_copy[i])
            scores.append(score)
    
#     bboxes = torch.Tensor(bboxes).unsqueeze(dim=0)
    labels = torch.Tensor(labels)
    scores = torch.Tensor(scores)

    del bboxes_copy, labels_copy, scores_copy

    return bboxes, labels, scores



def show_image_with_objects(image: np.array, 
                            bboxes: torch.Tensor, 
                            labels: torch.Tensor=None, 
                            scores: torch.Tensor=None,
                            threshold_score: float=0.5):

    image = Image.fromarray(image.transpose(1, 2, 0))

    if scores != None:
        bboxes, labels, scores = objects_threshold_scores(bboxes, labels, scores, threshold_score)

    for i in range(len(bboxes)):
        draw = ImageDraw.Draw(image)
        draw.rectangle(bboxes[i].numpy(), outline = color[labels[i].int()], width=2)

        if scores != None:
            bbox = draw.textbbox((bboxes[i][0], bboxes[i][1]), f"ID{int(labels[i])} {scores[i] * 100:.2f}%")
            draw.rectangle((bbox[0]-2, bbox[1]-2, bbox[2]+2, bbox[3]+2), fill=(0, 0, 0))
            draw.text((bboxes[i][0], bboxes[i][1]), f"ID{int(labels[i])} {scores[i] * 100:.2f}%", color[labels[i].int()])
        else:
            bbox = draw.textbbox((bboxes[i][0], bboxes[i][1]), f"ID{int(labels[i])}")
            draw.rectangle((bbox[0]-2, bbox[1]-2, bbox[2]+2, bbox[3]+2), fill=(0, 0, 0))
            draw.text((bboxes[i][0], bboxes[i][1]), f"ID{int(labels[i])}", color[labels[i]])
    return image


@torch.no_grad
def predict(model, image, confidence_threshold=0.3, overlap_threshold=0.3):
    """
    Forward pass through the model and get its predictions. 
    """

    with torch.no_grad():
        model.eval()
        model.to(DEVICE)
        outputs = model.predict(torch.Tensor(image).unsqueeze(dim=0).to(DEVICE))
    

    bboxes, labels, scores = objects_threshold_scores(outputs[0]['boxes'].to('cpu'), outputs[0]['labels'], outputs[0]['scores'], confidence_threshold)

    return bboxes, labels, scores


# # This sidebar UI lets the user select parameters for the object detector.
# def object_detector_ui():
#     st.sidebar.markdown("# Model")
#     confidence_threshold = st.sidebar.slider(
#         "Confidence threshold", 0.0, 1.0, 0.5, 0.01
#     )
#     overlap_threshold = st.sidebar.slider("Overlap threshold", 0.0, 1.0, 0.3, 0.01)
#     return confidence_threshold, overlap_threshold


# def object_selector_ui():
#     st.sidebar.markdown("# Objects to detect")
#     # The user can pick which type of object to search for.
#     object_type = st.sidebar.selectbox(
#         "Search for which objects?", OBJECTS_TO_DETECT[1:]
#     )
#     # The user can select a range for how many of the selected objecgt should be present.
#     min_objs, max_objs = st.sidebar.slider(
#         "How many %s s (select a range)?" % object_type, 0, 10, [3, 5]
#     )

#     return object_type, min_objs, max_objs


if __name__ == "__main__":
    
    model = load_model(MODEL_PATH)

    st.write("### [by mfclabber](https://github.com/mfclabber)")
    st.write("### ITMO University")

    readme_text = st.markdown(get_file_content_as_string("INFO_APP.md"))
    color = list((random.randint(40, 240), random.randint(40, 255), random.randint(60, 255)) for i in range(129))

    st.sidebar.title("What to do?")

    app_mode = st.sidebar.selectbox(
        "Choose the app mode", ["About the App", "Run the app", "Show the source code"]
    )

    if app_mode == "About the App":
        st.image("./assets/sample_7.png")
        st.sidebar.success('To continue select "Run the app".')

    elif app_mode == "Show the source code":
        readme_text.empty()
        st.write("You can find source code from my [GitHub](https://github.com/mfclabber/russian-food-recognition)")

    elif app_mode == "Run the app":
        readme_text.empty()

        st.write("# Running the object detection App")
        col1, col2, col3 = st.columns(3)

        with col1:
            st.write(' ')

        with col2:
            st.image("logo.jpeg")

        with col3:
            st.write(' ')

        st.write("## To load a sample image just click on the load sample image")

        if st.button("Load a sample Image"):
            random_image_path = random.choice(SAMPLE_IMAGES)
            image = np.array(Image.open(random_image_path).convert('RGB')).transpose(2, 0, 1)
            st.image(image.transpose(1, 2, 0), caption="Original image")
 
            bboxes, labels, scores = predict(model, torch.Tensor(image))
            bboxes, labels, scores = objects_threshold_scores(bboxes.to('cpu'), labels, scores, 0.4)

            image_new = show_image_with_objects(image, bboxes, labels, scores, 0.3)

            st.image(image_new, caption="Predicting labels on image")

        st.write("## Upload an Image to get its predictions")

        img_file_buffer = st.file_uploader("", type=["png", "jpg", "jpeg"])
        if img_file_buffer is not None:
            image = load_image_file(img_file_buffer)
            if image is not None:
                st.image(
                    image,
                    caption=f"Your image has shape {image.shape[0:2]}",
                )

                image = np.array(Image.open(img_file_buffer).convert('RGB')).transpose(2, 0, 1)
    
                bboxes, labels, scores = predict(model, torch.Tensor(image))
                bboxes, labels, scores = objects_threshold_scores(bboxes.to('cpu'), labels, scores, 0.4)

                image_new = show_image_with_objects(image, bboxes, labels, scores, 0.3)

                st.image(image_new, caption="Predicting labels on image")
                
            else:
                st.write("### INVALID INPUT")

        # st.image(image)
        # Load Pytorch model here. You can come here automatically if you have downloaded pt file itself.
        # model = load_model(MODEL_PATH)

    #     if flag == 1:
    #         image_out, labels, scores = predict(
    #             model, image, confidence_threshold, overlap_threshold
    #         )
    #         if len(labels) == 0:
    #             st.write("No relevant object detected in the image")
    #         else:
    #             st.image(image_out, use_column_width=True)
    #             st.write("- Image with detection")
    #             for i in range(len(labels)):
    #                 if OBJECTS_TO_DETECT[labels[i]] == object_type:
    #                     st.write("Successfully Detected object {}".format(object_type))
    #                     chk_fg = 1
    #                 st.write(
    #                     "Detected %s, with confidence %0.2f"
    #                     % (OBJECTS_TO_DETECT[labels[i]], scores[i])
    #                 )

    #             if chk_fg == 1:
    #                 st.write("Detected the required object: {}".format(object_type))