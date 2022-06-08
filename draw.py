import streamlit as st
from streamlit_drawable_canvas import st_canvas

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

# Do something interesting with the image data
if canvas_result.image_data is not None:
    st.image(canvas_result.image_data)