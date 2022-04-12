
import streamlit as st
import base64
from pathlib import Path
st.set_page_config(
    page_title="Car Damage Detector",
    page_icon="🚗",
    layout="wide",
    menu_items={
         'Get Help': 'https://www.linkedin.com/in/tridib-roy-974374145/',
         'Report a bug': "https://www.linkedin.com/in/tridib-roy-974374145/",
         'About': "Portfolio WebApp"
     }
)

st.title("Car Damage Detector")
# st.image("https://media.giphy.com/media/3o6MbhQZGGeskpDJLi/giphy.gif")
with st.expander("Expand for details on the classification model!!"):
    st.info("__Description:__ This model classifies the location & severity of damage on a car.")
    st.info("__Framework / model used:__ This model uses Detectron2, which is Facebook AI Research's next generation library that provides state-of-the-art detection and segmentation algorithms. \n" 
    "It is the successor of Detectron and maskrcnn-benchmark. It supports a number of computer vision research projects and production applications in Facebook.")
    st.image("https://user-images.githubusercontent.com/1381301/66535560-d3422200-eace-11e9-9123-5535d469db19.png")
    st.info("__Dataset used:__ It is trained on a custom dataset of car images which was manually annotated using VGG Image Annotator (VIA).")
    

name_cols=st.columns(2)
car_url= name_cols[0].text_input("Insert an url to check car damage: ")
try:
  st.image(car_url,caption="Uploaded image")
  with st.spinner("Processing the image and loading necessary files....."):
    import Detector
    data = Detector.car_damage_detector(car_url)
    parts = data[0]
    extent = data[1]
    st.success("Processing Completed!")
    st.write("")
    st.write("")
    st.info("The model classification results are as follows:  ")
    st.write(f"- Damaged Part detected close to {parts} area \n- The detected area seems to have {extent}\n\n")
    st.image("car_damage.jpeg",caption="The Classified Damages on the Car")
except:
  st.text("Waiting for image....")