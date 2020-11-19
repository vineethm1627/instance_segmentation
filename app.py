import streamlit as st
import cv2
import matplotlib.pyplot as plt
import pixellib
from pixellib.semantic import semantic_segmentation
from prediction import *

st.title("Instance Image Segmentation")
st.sidebar.title("Instance Image Segmentation")
st.sidebar.markdown("Mask R-CNN model trained on coco dataset for Instance Segmentation using PixelLib. Here, Different instances of the same object are segmented with the different color maps.")
pic = plt.imread("coco_dataset.png")
st.sidebar.image(pic, caption = "Model classes", use_column_width = True)
st.set_option('deprecation.showfileUploaderEncoding', False)
img_file = st.file_uploader("Upload the input image : ", type = ['jpg', 'jpeg', 'png'])

if img_file is not None:
    img = plt.imread(img_file, 0)
    st.image(img, caption = "Input Image", use_column_width = True)
    cv2.imwrite("input_images/in.jpg", img)

    out, out_box = prediction("input_images/in.jpg")
    
    col1, col2 = st.beta_columns(2)
    col1.image(out, caption = "Segmented Image", use_column_width = True)
    col2.image(out_box, caption = "Bounding Box", use_column_width = True)
