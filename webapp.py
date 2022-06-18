from turtle import width
import streamlit as st
from streamlit_drawable_canvas import st_canvas
import cv2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from piction import WriteFunc1,getSinRegression,plotRegression,getPoint,getDetedge

if 'flag' not in st.session_state:
  st.session_state["flag"] = 0



st.title("Drawable Canvas")
st.markdown("""
Draw on the canvas, get the image data back into Python !
* Doubleclick to remove the selected object when not in drawing mode
""")
st.sidebar.header("Configuration")

# Specify brush parameters and drawing mode
b_width = st.sidebar.slider("Brush width: ", 1, 100, 10)
b_color = st.sidebar.color_picker("Enter brush color hex: ")
bg_color = st.sidebar.color_picker("Enter background color hex: ", "#eee")
drawing_mode = st.sidebar.checkbox("Drawing mode ?", True)

# Create a canvas component
canvas_result  = st_canvas(
fill_color="rgba(255, 165, 0, 0.3)",
stroke_width=10,
stroke_color="Black",
background_color="White",
# width = 150,
height= 150,
key="canvas",
)

if st.button("関数", key=0):
  st.session_state["flag"] = 1

# Do something interesting with the image data
if canvas_result.image_data is not None and st.session_state["flag"]==1:
	#st.image(canvas_result.image_data)
	x_data,y_data=getPoint(canvas_result.image_data)
	print(x_data)
	x,y=getSinRegression(x_data,y_data)
	print(x)
	fig= plt.figure("picture function")

	plt.plot(x,y,label="RegressionFunction")
	plt.xlabel("num")
	plt.ylabel("f")
	plt.legend()
	#plt.savefig("/content/drive/MyDrive/hackason/results/"+fileName+".png")
	# グリッド表示
	plt.grid()
	st.pyplot(fig)
	