import os
import shutil

from google_images_download import google_images_download

def dowload_matting_dataset(output_dir):

    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)

    response = google_images_download.googleimagesdownload()
    response.download({
        "keywords": "portrait transparent background",
        "color_type": "transparent",
        "size": "medium",
        "limit": 500,
        "output_directory": output_dir,
        "chromedriver": "/usr/local/bin/chromedriver"})

    response = google_images_download.googleimagesdownload()
    response.download({
        "keywords": "texture background",
        "color_type": "full-color",
        "size": "medium",
        "limit": 500,
        "output_directory": output_dir,
        "chromedriver": "/usr/local/bin/chromedriver"})

if __name__ == "__main__":
    output_dir = os.path.join("data", "matting")
    dowload_matting_dataset(output_dir)