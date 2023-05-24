# -*- coding: utf-8 -*-
"""
Big Data Projekt Frame
"""

from PIL import Image
import os

"""Database"""
def Connector():
    pass

def WriteDatabase():
    pass

def ReadDatabase():
    pass

"""Preperation"""
def Path_generator():#directory = "path/to/directory"):
    directory = "D:\Ablage\weather_image_recognition\dew"
    for filename in os.listdir(directory):
        # Check if the file is an image file
        if filename.endswith(".jpg") or filename.endswith(".jpeg") or filename.endswith(".png"):
            # Open the image file
            yield Image.open(os.path.join(directory, filename))

def Get_color_scheme(vectors):
    color_scheme = []
    # drop doubles
    for rgb in vectors:
        if rgb not in color_scheme:
            color_scheme.append(rgb)
        # 20 fache verkleinerung.
        # fragen ob in 5er schritten um noch kleiner zu bekommen
    return color_scheme

def GoogleAPI():
    pass

def Image_to_rgb(image):
    # Convert the image to RGB mode if it's not already in RGB mode
    image = image.convert("RGB")
    
    # Get the width and height of the image
    width, height = image.size
    
    # Create an empty list to store the vectors
    vectors = []
    
    # Loop through each pixel in the image
    for x in range(width):
        for y in range(height):
            # Get the RGB values of the pixel
            r, g, b = image.getpixel((x, y))
            # Append the RGB values to the list as a vector
            vectors.append([r, g, b])
    
    # return the list of vectors
    return vectors

def PreparationCNN():
    pass

def CNN():
    pass

def Full_Preperation():
    # Loop through each file in the directory
    image_generator = Path_generator()

    # Open the image file
    for image in image_generator:
        vectors = Image_to_rgb(image)
        color_scheme = Get_color_scheme(vectors)
        break
    

"""Code"""
def Input():
    pass

def KNN_color_scheme():
    pass

def KNN_API():
    pass

def Predict():
    pass

def Output():
    pass

