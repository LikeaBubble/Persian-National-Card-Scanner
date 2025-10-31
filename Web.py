import streamlit as st 
import Pipeline
from PIL import Image

col1,col2 = st.columns(2)
model = Pipeline.pipeline('nano','mobilenet_large')

enable = col1.checkbox("Enable camera")
picture = col1.camera_input("Take a picture", disabled=not enable)

picture = col1.file_uploader('Image')
if picture:
    picture = Image.open(picture)
    out,angle = model.execute(picture)
    
    txt = ""
    for k,v in out.items():
        txt += f"{k} : {v}"
        txt += "\n"
    col2.text(txt)