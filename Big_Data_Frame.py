# -*- coding: utf-8 -*-
"""
Big Data Projekt Frame
"""

from PIL import Image
from tqdm import tqdm
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
    color_scheme1 = []
    count_scheme1 = []
    color_scheme2 = []
    count_scheme2 = []
    
    # drop doubles
    for rgb in tqdm(vectors):
        if rgb not in color_scheme1:
            color_scheme1.append(rgb)
            count_scheme1.append(1)
        else:
            count_scheme1[color_scheme1.index(rgb)] += 1
    for rgb in tqdm(color_scheme1):
        count_value = count_scheme1[color_scheme1.index(rgb)]
        converted_rgb = []
        for value in rgb:
            val_rest = value % 5
            if val_rest >= 3:
                converted_rgb.append(value + (5 - val_rest)) 
            else:
                converted_rgb.append(value - val_rest)
        if converted_rgb not in color_scheme2:
            color_scheme2.append(converted_rgb)
            count_scheme2.append(count_value)
        else:
            count_scheme2[color_scheme2.index(converted_rgb)] += count_value
    return color_scheme2, count_scheme2

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
        color_scheme, count_scheme = Get_color_scheme(vectors)
    

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
