import streamlit as st
import os
from ultralytics import YOLO

st.write("<h1 style='text-align: center';>Gym Back Machine Detector</h1>", unsafe_allow_html=True)
colA, colB, colC = st.columns(3)
st.image("./media/ezgif-5-81ce1448c1.gif", use_column_width=True)  # Display image in the column
st.write("Possible classes:")

# Displaying the possible classes as a JSON block
classes_dict = {"Class 0": "machine_seated_mid_row",
                "Class 1": "seated_cable_row",
                "Class 2": "cable_lat_pulldown",
                "Class 3": "machine_lat_pulldown",
                "Class 4": "chest_supported_T_bar_row",
                "Class 5": "landmine_T-bar_row",
                "Class 6": "machine_reverse_fly"}
st.json(classes_dict)
img = st.file_uploader("Please upload a back gym machine", type=("JPG", "PNG", "JPEG", "webp"))

if img:
    # Display image
    disp_image = st.empty()
    disp_image.image(img, use_column_width=True)

    # Save the uploaded image to a temporary file
    temp_file = "./temp_image.jpg"
    with open(temp_file, "wb") as f:
        f.write(img.getvalue())

    # Create empty placeholder for spinner in col2
    spinner = st.empty()

    with st.spinner("Predicting..."):
        model = YOLO('runs/detect/train/weights/best.pt')
        results = model.predict(source=temp_file, save=True)
        result_img_path = "runs/detect/predict/temp_image.jpg"

    # Clear the spinner and display results in col2
    spinner.empty()
    disp_image.image(result_img_path, use_column_width=True)

    # Remove the temporary file
    os.remove(temp_file)
    os.system('rm -rf runs/detect/predict')
    # os.remove(temp_file)
