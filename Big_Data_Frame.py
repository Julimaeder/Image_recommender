# -*- coding: utf-8 -*-
"""
Big Data Projekt Frame
"""
from PIL import Image
import numpy as np
from tqdm import tqdm
import os
import matplotlib.pyplot as plt
import mysql.connector


mydb = mysql.connector.connect(
  host="localhost",
  user="root",
  password="Informatiker1",
  autocommit=True
)

mycursor = mydb.cursor(buffered=True)

"""Database"""
def CreateDatabase():
    #just execute it if the Database isn't implemented
    mycursor.execute("create database bigData")
    mycursor.execute("create table images(ID varchar2(24) primary key not null, path varchar2(120))")
    mycursor.execute("create table pixels(ID varchar2(24), row int(100), col int(100), r int(3), g int(3), b int(3), primary key(ID, row, col), foreign key (ID) references images(ID))")
    mycursor.execute("create table schemes(ID varchar2(24), r int(3), g int(3), b int(3), amount int(10), primary key(ID, r, g, b), foreign key (ID) references images(ID))")

def LoadDatabase():
    #just execute it if the Database isn't implemented
    mycursor.execute("use bigData")

def CheckDatabase():
    #just execute it if the Database isn't implemented
    #mycursor.execute(
    sql_query = "SELECT IF(EXISTS (SELECT SCHEMA_NAME FROM INFORMATION_SCHEMA.SCHEMATA WHERE SCHEMA_NAME = 'bigData'), TRUE, FALSE) AS database_exists"
    result = pd.read_sql(sql_query, con=mydb)
    print(result)
    if result:
        LoadDatabase()
    else:
        CreateDatabase()

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

def Get_color_scheme_test(vectors):
    vectors = np.array(vectors)
    unique_vectors, vector_counts = np.unique(vectors, axis=0, return_counts=True)
    
    converted_vectors = np.where((unique_vectors % 5) >= 3, unique_vectors + (5 - (unique_vectors % 5)), unique_vectors - (unique_vectors % 5))
    
    unique_converted, converted_counts = np.unique(converted_vectors, axis=0, return_counts=True)
    
    return unique_converted.tolist(), converted_counts.tolist()

# def get_color_scheme_numpy(image):
#     # Get the pixel data from the image
#     pixels = np.array(image)

#     # Flatten the pixel data to a 2D array of shape (N, 3), where N is the total number of pixels
#     pixels_flat = pixels.reshape(-1, 3)

#     # Calculate the unique colors and their counts
#     unique_colors, color_counts = np.unique(pixels_flat, axis=0, return_counts=True)
    
#     color_scheme1 = list(unique_colors)
#     count_scheme1 = list(color_counts)
#     color_scheme2 = []
#     count_scheme2 = []

#     # Print the color counts
#     #for color, count in zip(unique_colors, color_counts):
#      #   print(f"RGB: {color}, Count: {count}")
#     for rgb in tqdm(color_scheme1):
#         count_value = count_scheme1[color_scheme1.index(rgb)]
#         converted_rgb = []
#         for value in rgb:
#             val_rest = value % 5
#             if val_rest >= 3:
#                 converted_rgb.append(value + (5 - val_rest)) 
#             else:
#                 converted_rgb.append(value - val_rest)
#         if converted_rgb not in color_scheme2:
#             color_scheme2.append(converted_rgb)
#             count_scheme2.append(count_value)
#         else:
#             count_scheme2[color_scheme2.index(converted_rgb)] += count_value
#     return color_scheme2, count_scheme2

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
        color_schemet, count_schemet = Get_color_scheme_test(vectors)
        #color_schemenp, count_schemenp = get_color_scheme_numpy(vectors)
        print(np.setdiff1d(np.array([color_scheme, count_scheme]), np.array([color_schemet, count_schemet]))) 
        print("done next")
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

Full_Preperation()