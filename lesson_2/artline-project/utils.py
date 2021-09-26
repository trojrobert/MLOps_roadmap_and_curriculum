import PIL.Image
import requests
from io import BytesIO
import os
import re
import base64
from fastai.vision import open_image
from matplotlib import pyplot as plt


def add_margin(pil_img, top, right, bottom, left, color):
    width, height = pil_img.size
    new_width = width + right + left
    new_height = height + top + bottom
    result = PIL.Image.new(pil_img.mode, (new_width, new_height), color)
    result.paste(pil_img, (left, top))
    return result


def visualize_image(prediction):
    plt.imshow(prediction)
    plt.show()


def parse_base64(string_):
    base64_path = "data:image/jpeg;base64,"
    if string_.startswith(base64_path):
        string_ = re.sub(base64_path, "", string_)
        string_ =  bytes(string_, "UTF-8")
        return base64.b64decode(string_)
    else:
        return None

def preprocess_img(img_url):
    bytes_str = parse_base64(img_url)
    if not bytes_str:
        res = requests.get(img_url)
        if res.status_code != 200:
            try:
                res.raise_for_status()
            except Exception as e:
                return str(e)
        else:
            bytes_io = BytesIO(res.content)
    else:
        bytes_io = BytesIO(bytes_str)
        
    img = PIL.Image.open(bytes_io).convert("RGB")
    img_path = os.path.join("static/images", "query_img.jpg")
    img.save(img_path)
    img = add_margin(img, 250, 250, 250, 250, (255, 255, 255))
    img.save(img_path, quality=95)
    img = open_image(img_path)
    return img
            