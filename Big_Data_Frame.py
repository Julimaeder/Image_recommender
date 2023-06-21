# -*- coding: utf-8 -*-
"""
Big Data Projekt Frame
"""
from PIL import Image
import numpy as np
import pandas as pd
from tqdm import tqdm
import os
import pandas as pd
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
    mycursor.execute("use bigData")
    mycursor.execute("create table images(iID int primary key not null, ipath varchar(120) unique)")
    mycursor.execute("create table pixels(pID int, prow int(100), pcol int(100), r int(3), g int(3), b int(3), primary key(pID, prow, pcol), foreign key (pID) references images(iID))")
    mycursor.execute("create table schemes(sID int, r int(3), g int(3), b int(3), amount int(10), primary key(sID, r, g, b), foreign key (sID) references images(iID))")
    mydb.commit()

def LoadDatabase():
    #just execute it if the Database isn't implemented
    mycursor.execute("use bigData")

def CheckDatabase():
    #just execute it if the Database isn't implemented
    #mycursor.execute(
    sql_query = "SELECT IF(EXISTS (SELECT SCHEMA_NAME FROM INFORMATION_SCHEMA.SCHEMATA WHERE SCHEMA_NAME = 'bigData'), TRUE, FALSE) AS database_exists"
    result = pd.read_sql(sql_query, con=mydb)
    print(str(result["database_exists"].item()))
    if str(result["database_exists"].item()) != "0":
        LoadDatabase()
    else:
        CreateDatabase()

def ReadIDbyPath(path):
    sql_query = f'SELECT iID FROM images where ipath="{path}"'
    result = pd.read_sql(sql_query, con=mydb)
    return int(result["iID"].item())

def imageMaxID():
    sql_query = 'SELECT MAX(iID) as vali FROM images'
    result = pd.read_sql(sql_query, con=mydb)
    return result["vali"].item()

def writeImage(id,path):
    mycursor.execute(f'insert into images values({id}, "{path}") ON DUPLICATE KEY UPDATE ipath="{path}"')
    mydb.commit()

def writePixels(id,row,col,r,g,b):
    mycursor.execute(f'insert into pixels values({id}, {row}, {col}, {r}, {g}, {b}) ON DUPLICATE KEY UPDATE r={r}, g={g}, r={b}')
    mydb.commit()

def writeSchemes(id,r,g,b,amount):
    mycursor.execute(f'insert into schemes values({id}, {r}, {g}, {b}, {amount}) ON DUPLICATE KEY UPDATE amount={amount}')
    mydb.commit()

"""Preperation"""
def Path_generator():#directory = "path/to/directory"):
    directory = "D:\Ablage\weather_image_recognition"
    for root, dirs, files in os.walk(directory):
        for filename in files:
            # Check if the file is an image file
            if filename.endswith(".jpg") or filename.endswith(".jpeg") or filename.endswith(".png"):
                # Open the image file
                yield os.path.join(root, filename)


def Get_color_scheme(vectors):
    vectors = np.array(vectors)
    unique_vectors, vector_counts = np.unique(vectors, axis=0, return_counts=True)
    
    converted_vectors = np.where((unique_vectors % 5) >= 3, unique_vectors + (5 - (unique_vectors % 5)), unique_vectors - (unique_vectors % 5))
    
    unique_converted, converted_counts = np.unique(converted_vectors, axis=0, return_counts=True)
    
    return unique_converted, converted_counts

def GoogleAPI():
    pass

def Image_to_rgb_scheme(image):
    # Convert the image to RGB mode if it's not already in RGB mode
    image = image.convert("RGB")
    
    # Get the width and height of the image
    width, height = image.size
    im = np.array(image)
    return im

def Image_to_rgb_image(image, image_id):
    # Convert the image to RGB mode if it's not already in RGB mode
    image = image.convert("RGB")
    
    # Get the width and height of the image
    width, height = image.size
    
    # Loop through each pixel in the image
    for x in range(width):
        for y in range(height):
            # Get the RGB values of the pixel
            r, g, b = image.getpixel((x, y))
            # Append the RGB values to the list as a vector
            # korrekturbedarf
            print(image_id, x, y, r, g, b)
            writePixels(image_id, x, y, r, g, b)

def PreparationCNN():
    pass

def CNN():
    pass

def Full_Preperation():
    # Loop through each file in the directory
    image_generator = Path_generator()

    maxID = imageMaxID()
    counter = 0

    if maxID is None:
        print("all good")
    else:
        counter = int(maxID)

    # Open the image file
    for image_path in image_generator:
        counter += 1

        print(image_path)
        writeImage(str(counter), image_path)

        image_id = ReadIDbyPath(image_path)

        image = Image.open(image_path)
        vectors = Image_to_rgb_scheme(image)

        color_scheme, count_scheme = Get_color_scheme(vectors)
        #color_schemenp, count_schemenp = get_color_scheme_numpy(vectors)
        for cols, couns in zip(color_scheme, count_scheme):
            print(image_id, cols[0], cols[1], cols[2], couns)
            writeSchemes(image_id, cols[0], cols[1], cols[2], couns)
        #print(color_scheme, "\n", count_scheme)
         
        print(len(vectors))
        image = image.resize((400,400))
        Image_to_rgb_image(image, image_id)
        print(400*400)
        
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

def myMain():
    CheckDatabase()
    Full_Preperation()



print("Start")
myMain()
print("end")
mycursor.close()
mydb.close()