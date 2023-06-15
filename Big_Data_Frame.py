# -*- coding: utf-8 -*-
"""
Big Data Projekt Frame
"""
from PIL import Image
import numpy as np
import pandas as pd
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
    directory = "C:\Shazil Khan\HSD\MachinePerception\img"
    for filename in os.listdir(directory):
        # Check if the file is an image file
        if filename.endswith(".jpg") or filename.endswith(".jpeg") or filename.endswith(".png"):
            # Open the image file
            yield Image.open(os.path.join(directory, filename))


def Get_color_scheme_test(vectors):
    print(type(vectors))
    vectors = vectors.reshape(-1,3)
    unique_vectors, vector_counts = np.unique(vectors, axis=0, return_counts=True)
    
    converted_vectors = np.where((unique_vectors % 5) >= 3, unique_vectors + (5 - (unique_vectors % 5)), unique_vectors - (unique_vectors % 5))
    
    unique_converted, converted_counts = np.unique(converted_vectors, axis=0, return_counts=True)
    
    return unique_converted, converted_counts

def GoogleAPI():
    pass

def Image_to_rgb(image):
    # Convert the image to RGB mode if it's not already in RGB mode
    image = image.convert("RGB")
    
    # Get the width and height of the image
    width, height = image.size
    im = np.array(image)
    return im

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
        color_scheme, count_scheme = Get_color_scheme_test(vectors)
        print(color_scheme.shape, count_scheme.shape)
        print(color_scheme, count_scheme)
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