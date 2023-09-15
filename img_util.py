import base64
from PIL import Image
from io import BytesIO
import numpy as np
import cv2

def base64_img(base64str):
    return Image.open(BytesIO(base64.b64decode(base64str)))

def img_base64(image):
    buffered = BytesIO()
    image.save(buffered, format="JPEG")
    img_str = base64.b64encode(buffered.getvalue())
    return img_str

def base64_cv(base64str):
    im_bytes = base64.b64decode(base64str)
    im_arr = np.frombuffer(im_bytes, dtype=np.uint8)  # im_arr is one-dim Numpy array
    img = cv2.imdecode(im_arr, flags=cv2.IMREAD_COLOR)
    return img

def resize_image(img, scale_factor=0.5):
    # 打开图片
    # img = Image.open(input_path).convert("RGB")
    
    # 获取原始图片的宽度和高度
    original_width, original_height = img.size
    scale_factor = min(960/max(original_width,original_height),
                       540/min(original_width,original_height),
    )
    
    # 计算缩放后的新宽度和新高度
    new_width = int(original_width * scale_factor)
    new_height = int(original_height * scale_factor)
    
    # 等比例缩放图片
    resized_img = img.resize((new_width, new_height), Image.LANCZOS)
    return resized_img


