from PIL import Image, ImageFile
from tqdm import tqdm
import numpy as np
import pandas as pd
import os
import sqlite3
ImageFile.LOAD_TRUNCATED_IMAGES = True

sqlPath = "./Data/databases/my_database.db"
imagesPath = "E:\\images\\"
predictPath = "D:\\Ablage\\predict"

def CreateDatabase(mydb,mycursor):
    # just execute it if the Database isn't implemented
    mycursor.execute("CREATE TABLE IF NOT EXISTS images(iID INT PRIMARY KEY NOT NULL, ipath VARCHAR(120) UNIQUE);")
    mydb.commit()
    mycursor.execute("CREATE TABLE IF NOT EXISTS schemes(sID INT PRIMARY KEY, FOREIGN KEY (sID) REFERENCES images(iID));")
    mydb.commit()
    anzahl_spalten = 216
    for spalten_nr in range(0, anzahl_spalten):
        spaltenname = f"spalte_{spalten_nr}"
        mycursor.execute(f"ALTER TABLE schemes ADD {spaltenname} int")
    mydb.commit()
    mycursor.close()
    mydb.close()
    
def CheckDatabase():
    mydb = sqlite3.connect(sqlPath)
    mycursor = mydb.cursor()
    # check if the Database is implemented
    sql_query = 'SELECT name FROM sqlite_master WHERE name="schemes"'
    result = pd.read_sql(sql_query, con=mydb)
    if len(result['name']) == 0:
        CreateDatabase(mydb,mycursor)
    print("connected")
    mycursor.close()
    mydb.close()

def writeSchemes(df):
    mydb = sqlite3.connect(sqlPath)
    mycursor = mydb.cursor()
    df.to_sql(name='schemes', con=mydb, if_exists='append',index = False, chunksize = 1000)
    mycursor.close()
    mydb.close()

def ReadIDbyPath(path):
    mydb = sqlite3.connect(sqlPath)
    mycursor = mydb.cursor()
    sql_query = f'SELECT iID FROM images where ipath="{path}"'
    result = pd.read_sql(sql_query, con=mydb)
    mycursor.close()
    mydb.close()
    if len(result["iID"]) > 0:
        return int(result["iID"].item())
    else:
        print("klappt nicht")
        return None

def readAllImages():
    mydb = sqlite3.connect(sqlPath)
    mycursor = mydb.cursor()
    sql_query = 'SELECT * FROM images INNER JOIN schemes ON images.iID = schemes.sID'
    result = pd.read_sql(sql_query, con=mydb)
    mycursor.close()
    mydb.close()
    if len(result["iID"]) > 0:
        return result
    else:
        return None

def ReadPathbyID(id):
    mydb = sqlite3.connect(sqlPath)
    mycursor = mydb.cursor()
    sql_query = f'SELECT ipath FROM images where iID="{id}"'
    result = pd.read_sql(sql_query, con=mydb)
    print(result)
    mycursor.close()
    mydb.close()
    if len(result["ipath"]) > 0:
        return result["ipath"].item()
    else:
        return None
    
def ReadALLID():
    mydb = sqlite3.connect(sqlPath)
    mycursor = mydb.cursor()
    sql_query = 'SELECT iID FROM images'
    result = pd.read_sql(sql_query, con=mydb)
    mycursor.close()
    mydb.close()
    if len(result["iID"]) > 0:
        return list(result["iID"])
    else:
        return None
    
def ReadSchemesbyID(id):
    mydb = sqlite3.connect(sqlPath)
    mycursor = mydb.cursor()
    sql_query = f'SELECT * FROM schemes where sID="{id}"'
    result = pd.read_sql(sql_query, con=mydb)
    mycursor.close()
    mydb.close()
    if len(result['sID']) > 0:
        return result
    else:
        return None

def imageMaxID():
    mydb = sqlite3.connect(sqlPath)
    mycursor = mydb.cursor()
    sql_query = 'SELECT MAX(iID) as vali FROM images'
    result = pd.read_sql(sql_query, con=mydb)
    mycursor.close()
    mydb.close()
    if len(result["vali"]) > 0:
        return result["vali"].item()
    else:
        return None

def writeImage(id,path):
    mydb = sqlite3.connect(sqlPath)
    mycursor = mydb.cursor()
    mycursor.execute(f'insert into images values({id}, "{path}")')
    mydb.commit()
    mycursor.close()
    mydb.close()

"""Preperation"""
def Path_generator(ipath):
    directory = ipath
    for root, dirs, files in os.walk(directory):
        for filename in files:
            # Check if the file is an image file
            if filename.endswith(".jpg") or filename.endswith(".jpeg") or filename.endswith(".png"):
                # Open the image file
                yield os.path.join(root, filename)

def Get_color_scheme(vectors):
    new_vectors = np.array(vectors)
    u_vectors = new_vectors - (new_vectors % 51)
    unique_vectors, unique_counts = np.unique(u_vectors, axis=0, return_counts=True)
    uvs = unique_vectors / 51
    return uvs, unique_counts

def Image_to_rgb_scheme(image):
    # Convert the image to RGB mode if it's not already in RGB mode
    image = image.convert("RGB")
    size = 300
    # Get the width and height of the image
    width, height = image.size
    if width > size and height > size:
        image = image.resize((size, size))
    elif height > size:
        image = image.resize((width, size))
    elif width > size:
        image = image.resize((size, height))
    im3d = np.array(image)
    im = im3d.reshape(-1, im3d.shape[-1])
    return im

def Full_Preperation():
    CheckDatabase()
    # Loop through each file in the directory
    image_generator = Path_generator(imagesPath)
    maxID = imageMaxID()
    counter = 0
    if maxID is None:
        print("all good")
    else:
        counter = int(maxID)
    # Open the image file
    for image_path in tqdm(image_generator, total=140395):
        image_id = ReadIDbyPath(image_path)
        if image_id is None:
            counter += 1
            writeImage(counter, image_path)
            image = Image.open(image_path)
            vectors = Image_to_rgb_scheme(image)
            color_scheme, count_scheme = Get_color_scheme(vectors)
            rgb_values = np.zeros((6, 6, 6))
            rgb_values[color_scheme[:, 0].astype('int32'), color_scheme[:, 1].astype('int32'), color_scheme[:, 2].astype('int32')] = count_scheme
            num = rgb_values.flatten().astype('int32')
            values = num.tolist() 
            anzahl_spalten = 216
            cols = ["sID"]+ [f"spalte_{spalten_nr}" for spalten_nr in range(anzahl_spalten)]
            data = [counter] + values
            df = pd.DataFrame(columns=cols)
            df.loc[0] = data
            # Insert the data into the database
            writeSchemes(df)
    
"""Code"""
def Distances(image, df1, num_images=5):
    distances = []
    for i in tqdm(range(len(df1))):
        colors = df1.iloc[i].values[1:]
        distance = np.linalg.norm(image - colors)
        distances.append(distance)
    paths = df1.iloc[:,0].tolist()
    df2 = pd.DataFrame({'path': paths,'enum': distances})
    nearest_images = df2.nsmallest(num_images, "enum")
    return nearest_images

def Full_Prediction(image_path, df):
    df1= df.drop(['sID', 'iID'], axis=1)
    # Open and load all images
    image = Image.open(image_path)  
    vectors = Image_to_rgb_scheme(image)
    color_scheme, count_scheme = Get_color_scheme(vectors)
    rgb_values = np.zeros((6, 6, 6), dtype=int)
    rgb_values[color_scheme[:, 0].astype('int32'), color_scheme[:, 1].astype('int32'), color_scheme[:, 2].astype('int32')] = count_scheme
    np_i = rgb_values.flatten().astype('int32')
    color_scheme_distances = Distances(np_i, df1)
    smallest_list = []
    for path in color_scheme_distances["path"]:
        smallest_list.append(path)
    return smallest_list
