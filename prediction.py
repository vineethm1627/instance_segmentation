import os
import pixellib
import matplotlib.pyplot as plt
from pixellib.instance import instance_segmentation
DATA_URL = "mask_rcnn_coco.h5"

def prediction(img_file):
    # instantiating the instance segmentation class
    segment_image = instance_segmentation()

    # loading the model mask rcnn trained on coco dataset
    segment_image.load_model(DATA_URL)

    # performing the segmentation on the input image
    segment_image.segmentImage(img_file, output_image_name = "output_images/out.jpg")
    out = plt.imread("output_images/out.jpg", 0)

    # performing the segmentation on the input image with bounding boxes
    segment_image.segmentImage(img_file, output_image_name = "output_images/out_box.jpg", show_bboxes = True)
    out_box = plt.imread("output_images/out_box.jpg", 0)

    return out, out_box
