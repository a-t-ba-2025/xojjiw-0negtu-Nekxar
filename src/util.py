import shutil
import cv2
import matplotlib.pyplot as plt
import numpy as np
import os
import pytesseract
import json
from dotenv import load_dotenv

load_dotenv()


def save_file(text_image, save_dir, filename):
    pdf_path = os.path.join(os.getenv(save_dir), f"{filename}.pdf")
    os.makedirs(os.path.dirname(pdf_path), exist_ok=True)
    with open(pdf_path, "wb") as f:
        f.write(text_image.read())


def save_image(image, save_dir, filename=None):
    save_direction = os.getenv(save_dir)
    os.makedirs(save_direction, exist_ok=True)

    valid_extensions = [".png", ".jpg", ".jpeg"]
    name, ext = os.path.splitext(filename)

    # Add default extension if none or invalid
    if ext.lower() not in valid_extensions:
        filename = name + ".png"

    save_path = os.path.join(save_direction, filename)

    # Check if image is a valid numpy array
    if not isinstance(image, np.ndarray):
        print(f"[WARNING] Cannot save image: expected numpy array but got {type(image)}. Skipping save.")
        return

    # Convert RGB to BGR for OpenCV if needed
    if len(image.shape) == 3 and image.shape[2] == 3:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    cv2.imwrite(save_path, image)


def show_image(images, titles=None, cmap="gray"):
    if isinstance(images, np.ndarray):
        images = [images]

    num_images = len(images)

    # Layout: max 6 images per row
    cols = min(6, num_images)
    rows = (num_images + cols - 1) // cols

    # Default titles if none provided
    if titles is None:
        titles = [f"Image {i + 1}" for i in range(num_images)]

    # Create larger figure: each image gets more space
    fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 6 * rows))
    axes = np.array(axes).reshape(-1)  # Flatten in case of single row/col

    for idx, img in enumerate(images):
        ax = axes[idx]

        # If RGB image, convert from BGR to RGB
        if len(img.shape) == 3 and img.shape[2] == 3:
            img_display = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            ax.imshow(img_display)
        else:
            ax.imshow(img, cmap=cmap)

        ax.set_title(titles[idx], fontsize=14)
        ax.axis('off')

    # Hide unused subplots
    for i in range(len(images), len(axes)):
        axes[i].axis('off')

    plt.tight_layout()
    plt.show()


# Load Tesseract path from .env file
pytesseract.pytesseract.tesseract_cmd = os.getenv("TESSERACT_CMD")
def get_ocr_score(image):
    text = pytesseract.image_to_string(image, lang='deu')  # OCR in language German
    return len(text.strip())  # Remove spaces and count characters


def convert_numpy(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    return str(obj)  # Fallback for others


def save_json(results, save_dir, filename):
    save_direction = os.getenv(save_dir)
    os.makedirs(save_direction, exist_ok=True)

    base = os.path.splitext(os.path.basename(filename))[0]
    json_path = os.path.join(save_direction, base + ".json")

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False, default=convert_numpy)
