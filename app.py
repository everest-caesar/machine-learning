import streamlit as st
import cv2
from PIL import Image, ImageEnhance
import numpy as np
import os 

@st.cache
def load_image(img):
    im = Image.open(img)
    return im

Face_Cascade = cv2.CascadeClassifier("freecog/haarcascade_frontalface_default.xml")
Eyes_Cascade = cv2.CascadeClassifier("freecog/haarcascade_eye.xml")
Smiles_Cascade = cv2.CascadeClassifier("freecog/haarcascade_smile.xml")

def detect_faces(ourimage):
    new_img = np.array(ourimage.convert("RGB"))
    img = cv2.cvtColor(new_img, 1)
    gray = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
    faces = Face_Cascade.detectMultiScale(gray,1.1,4)
    for (x,y,w,h) in faces:
        cv2.rectangle(img, (x,y), (x+w,y+h), (255,0,0),2)
    return img,faces


def detect_eyes(ourimage):
    new_img = np.array(ourimage.convert("RGB"))
    img = cv2.cvtColor(new_img, 1)
    gray = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
    eyes = Eyes_Cascade.detectMultiScale(gray,1.1,4)
    for (ex,ey,ew,eh) in eyes:
        cv2.rectangle(img, (ex,ey), (ex+ew,ey+eh), (0,255,0),2)
    return img,eyes


def detect_smiles(ourimage):
    new_img = np.array(ourimage.convert("RGB"))
    img = cv2.cvtColor(new_img, 1)
    gray = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
    smiles = Smiles_Cascade.detectMultiScale(gray,1.1,4)
    for (x,y,w,h) in smiles:
        cv2.rectangle(img, (x,y), (x+w,y+h), (255,0,0),2)
    return img,smiles


def cartonize_image(ourimage):
    new_img = np.array(ourimage.convert("RGB"))
    img = cv2.cvtColor(new_img, 1)
    gray = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
    gray= cv2.medianBlur(gray, 5)
    edges = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 9, 9)
    color = cv2.bilateralFilter(img, 9, 300, 300)
    cartoon = cv2.bitwise_and(color, color, mask=edges)
    return cartoon


def cannize_image(ourimage):
    new_img = np.array(ourimage.convert("RGB"))
    img = cv2.cvtColor(new_img, 1)
    img = cv2.GaussianBlur(img, (11, 11), 0)
    canny = cv2.Canny(img, 100, 150)
    return canny





def main():
    st.title("face detection system")
    st.text("build with streamlit and opencv")

    Activities = ["detection","about", "help"]
    choice = st.sidebar.selectbox("select activity", Activities)


    if choice == "detection":
        st.subheader("Face detection")

        imagefile = st.file_uploader("Upload image", type=['jpg', 'png', 'jpeg'])
        if imagefile is not None:
            ourimage = Image.open(imagefile)
            #st.write(type(ourimage))
            st.text("OTIGINAL IMAGE UPLOAD")
            st.image(ourimage)

        enhancetype = st.sidebar.radio('Enhance Type',["Original", "Gray-Scale", "Contrast", "Brightness", "Blurring"])
        if enhancetype == "Gray-Scale":
            new_img = np.array(ourimage.convert('RGB'))
            img = cv2.cvtColor(new_img,1)
            gray = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
            #st.write(new_img)
            st.image(gray)

        if enhancetype == "Contrast":
            c_rate = st.sidebar.slider("Contrast",0.5,3.5)
            enhancer = ImageEnhance.Contrast(ourimage)
            imageoutput = enhancer.enhance(c_rate)
            #st.write(new_img)
            st.image(imageoutput)

        if enhancetype == "Brightness":
            c_rate = st.sidebar.slider("Brightness",0.5,3.5)
            enhancer = ImageEnhance.Brightness(ourimage)
            imageoutput = enhancer.enhance(c_rate)
            #st.write(new_img)
            st.image(imageoutput)

        if enhancetype == "Blurring":
            new_img = np.array(ourimage.convert('RGB'))
            blur_rate = st.sidebar.slider("Blurring",0.5,3.5)
            img = cv2.cvtColor(new_img,1)
            blur = cv2.GaussianBlur(img,(11,11), blur_rate)
            #st.write(new_img)
            st.image(blur)

        else:
            st.image(ourimage,width=300)


        task = ["Faces", "Smiles", "Eyes", "Cannize", "Cartonize"]
        feature_choice = st.sidebar.selectbox('Find Features', task)

        if st.button("Process"):
            if feature_choice == "Faces":
                result_img, result_faces = detect_faces(ourimage)
                st.image(result_img)
                st.success("Found {} faces".format(len(result_faces)))

            elif feature_choice == "Eyes":
                result_img, result_eyes = detect_eyes(ourimage)
                st.image(result_img)
                st.success("Found {} eyes".format(len(result_eyes)))

            elif feature_choice == "Smiles":
                result_img, result_smiles = detect_smiles(ourimage)
                st.image(result_img)
                st.success("Found {} smiles".format(len(result_smiles)))

            elif feature_choice == 'Cartonize':
                result_img = cartonize_image(ourimage)
                st.image(result_img)

            elif feature_choice == 'Cannize':
               result_canny = cannize_image(ourimage)
               st.image(result_canny)


            

        





    elif choice == "about":
        st.text("go and read more about face detection")

    elif choice == "help":
        st.text("contact me 09056760962")

if __name__== "__main__":
    main()


