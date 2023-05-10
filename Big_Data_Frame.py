# -*- coding: utf-8 -*-
"""
Big Data Projekt Frame
"""

from PIL import Image
import os

"""Datenbank"""
def connector():
    pass

def WriteDatenbank():
    pass

def ReadDatenbank():
    pass

"""Vorbereitung"""
def GetFarbschema():
    pass

def GoogleAPI():
    pass

def image_to_rgb():
    # Set the path to the directory that contains the images
    directory = "path/to/directory"

    # Loop through each file in the directory
    def get_pictures(directory):
        for filename in os.listdir(directory):
            # Check if the file is an image file
            if filename.endswith(".jpg") or filename.endswith(".jpeg") or filename.endswith(".png"):
                # Open the image file
                yield Image.open(os.path.join(directory, filename))

    image_generator = get_pictures(directory)

    # Open the image file
    for image in image_generator:
        
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
        yield vectors

def VorbereitungCNN():
    pass

def CNN():
    pass

"""Programm"""
def Eingabe():
    pass

def KNNFarbschema():
    pass

def KNNAPI():
    pass

def Predict():
    pass

def Ausgabe():
    pass

