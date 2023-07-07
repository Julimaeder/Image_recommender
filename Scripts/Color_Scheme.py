from PIL import Image, ImageFile
from tqdm import tqdm
import numpy as np
import pandas as pd
from tqdm import tqdm
import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import sqlite3
import time
import datetime
#import warnings

#warnings.simplefilter('ignore', category=NumbaDeprecationWarning)
#warnings.simplefilter('ignore', category=NumbaPendingDeprecationWarning)
#from sqlalchemy import create_engine
ImageFile.LOAD_TRUNCATED_IMAGES = True

sqlPath = "./Data/databases/my_database.db"
imagesPath = "E:\\images\\"
predictPath = "D:\\Ablage\\predict"


mydb = sqlite3.connect(sqlPath)
mycursor = mydb.cursor()



def CreateDatabase():
    # just execute it if the Database isn't implemented
    mycursor.execute("CREATE TABLE IF NOT EXISTS images(iID INT PRIMARY KEY NOT NULL, ipath VARCHAR(120) UNIQUE);")
    mydb.commit()
    #mycursor.execute("CREATE TABLE IF NOT EXISTS embeddings(eID INT, erow INT, ecol INT, r INT, g INT, b INT, PRIMARY KEY(eID, erow, ecol), FOREIGN KEY (eID) REFERENCES images(iID));")
    #mydb.commit()
    mycursor.execute("CREATE TABLE IF NOT EXISTS schemes(sID INT PRIMARY KEY, FOREIGN KEY (sID) REFERENCES images(iID));")
    mydb.commit()
    anzahl_spalten = 216
    for spalten_nr in range(0, anzahl_spalten):
        spaltenname = f"spalte_{spalten_nr}"
        mycursor.execute(f"ALTER TABLE schemes ADD {spaltenname} int")
    mydb.commit()
    


def CheckDatabase():
    # check if the Database is implemented
    sql_query = f'SELECT name FROM sqlite_master WHERE name="schemes"'
    result = pd.read_sql(sql_query, con=mydb)
    if len(result['name']) == 0:
        CreateDatabase()
    print("connected")


CheckDatabase()

#engine = create_engine('mysql+mysqldb://bigData:Informatiker1@localhost/root', echo = False)

def ReadIDbyPath(path):
    sql_query = f'SELECT iID FROM images where ipath="{path}"'
    result = pd.read_sql(sql_query, con=mydb)
    if len(result["iID"]) > 0:
        return int(result["iID"].item())
    else:
        return None

def readAllImages():
    sql_query = f'SELECT * FROM images INNER JOIN schemes ON images.iID = schemes.sID'
    result = pd.read_sql(sql_query, con=mydb)
    if len(result["iID"]) > 0:
        return result
    else:
        return None


def ReadPathbyID(id):
    sql_query = f'SELECT ipath FROM images where iID="{id}"'
    result = pd.read_sql(sql_query, con=mydb)
    print(result)
    if len(result["ipath"]) > 0:
        return result["ipath"].item()
    else:
        return None
    
def ReadALLID():
    sql_query = f'SELECT iID FROM images'
    result = pd.read_sql(sql_query, con=mydb)
    if len(result["iID"]) > 0:
        return list(result["iID"])
    else:
        return None
    
def ReadSchemesbyID(id):
    sql_query = f'SELECT * FROM schemes where sID="{id}"'
    result = pd.read_sql(sql_query, con=mydb)
    if len(result['sID']) > 0:
        return result
    else:
        return None

def imageMaxID():
    sql_query = 'SELECT MAX(iID) as vali FROM images'
    result = pd.read_sql(sql_query, con=mydb)
    if len(result["vali"]) > 0:
        return result["vali"].item()
    else:
        return None

def writeImage(id,path):
    mycursor.execute(f'insert into images values({id}, "{path}")')
    mydb.commit()


"""Preperation"""
def Path_generator(ipath):#directory = "path/to/directory"):
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

# @jit
#def Get_color_scheme(vectors):
#     vectors = np.array(vectors)
#     vector_color, vector_counts = np.unique(vectors, axis=0, return_counts=True)
#     u_vectors = vector_color - (vector_color%51)
#     unique_vectors, unique_counts = np.unique(u_vectors, axis=0, return_counts=True)
#     count = 0
#     numbers = np.zeros(len(unique_counts), dtype=np.int64)

#     for i, c in enumerate(unique_counts):
#         numbers[i] = np.sum(vector_counts[count:count+c])
#         count += c
#     uvs = unique_vectors / 51
#     return uvs, numbers


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
    # Loop through each file in the directory
    image_generator = Path_generator(imagesPath)
    
    print("hi")
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
            #t = time.time()
            counter += 1
            #print(counter)
            
            #time1 = time.time()
            writeImage(counter, image_path)
            #try:
            image = Image.open(image_path)

            vectors = Image_to_rgb_scheme(image)
            
            #print(vectors.shape)
            
            #time2 = time.time()
            color_scheme, count_scheme = Get_color_scheme(vectors)
            #print(color_scheme.shape, count_scheme.shape)

            
            #time3 = time.time()
            rgb_values = np.zeros((6, 6, 6))
            rgb_values[color_scheme[:, 0].astype('int32'), color_scheme[:, 1].astype('int32'), color_scheme[:, 2].astype('int32')] = count_scheme
            
            num = rgb_values.flatten().astype('int32')#time4 = time.time()
            
            values = num.tolist() 
            #num = rgb_values.transpose(2,0,1).reshape((8,-1)).astype('int32')
            #print(num.shape)

            anzahl_spalten = 216

            #df = pd.DataFrame({'sID': counter})
            cols = ["sID"]+ [f"spalte_{spalten_nr}" for spalten_nr in range(anzahl_spalten)]
            data = [counter] + values
            df = pd.DataFrame(columns=cols)
            df.loc[0] = data
            #time5 = time.time()
            # Merge the two DataFrames based on index
            #df = df.merge(column_data, left_index=True, right_index=True)     

            #df = df[df.iloc[:, 2:].sum(axis=1) != 0]
            #print(df.shape)
            

            #time6 = time.time()
            # Insert the data into the database
            df.to_sql(name='schemes', con=mydb, if_exists='append',index = False, chunksize = 1000)
                     
            #print("time: ", time.time()-t)
            #time7 = time.time()   
            #print("1-2:", time2-time1)
            #print("2-3:", time3-time2)
            #print("3-4:", time4-time3)
            #print("4-5:", time5-time4)
            #print("5-6:", time6-time5)
            #print("7-6:", time7-time6)
            #except :
                #print("continue")
            #print(color_scheme, "\n", count_scheme)

def predictschemes_gen(image):
    vectors = Image_to_rgb_scheme(image)
    color_scheme, count_scheme = Get_color_scheme(vectors)
    rgb_values = np.zeros((6, 6, 6), dtype=int)
    rgb_values[color_scheme[:, 0].astype('int32'), color_scheme[:, 1].astype('int32'), color_scheme[:, 2].astype('int32')] = count_scheme
        
    np_i = rgb_values.flatten().astype('int32').tolist()

    #print(color_scheme.shape, len(count_scheme))
    ids = ReadALLID()
    for j in ids:
        df_j = ReadSchemesbyID(j)
        if df_j is not None:
            df_j.values.tolist()
            #print(i,j)
            np_j = df_j.iloc[:, 1:].values.tolist()
            e = Distances(j, np_i,np_j)
            yield j, e 

    
"""Code"""

def Distances(image, df1, num_images=5):
    distances = []
    #nearest_images = []
    for i in tqdm(range(len(df1))):
        colors = df1.iloc[i].values[1:]
        distance = np.linalg.norm(image - colors)
        distances.append(distance)
    paths = df1.iloc[:,0].tolist()
    df2 = pd.DataFrame({'path': paths,'enum': distances})
    nearest_images = df2.nsmallest(num_images, "enum")
    
    return nearest_images


def Full_Prediction(image_path):

    df = readAllImages()
    df= df.drop(['sID', 'iID'], axis=1)
    print("Loaded Data")
    
    # Open and load all images
    image = Image.open(image_path)
        
    #schemes = predictschemes_gen(image)
    vectors = Image_to_rgb_scheme(image)

    color_scheme, count_scheme = Get_color_scheme(vectors)
    rgb_values = np.zeros((6, 6, 6), dtype=int)
    rgb_values[color_scheme[:, 0].astype('int32'), color_scheme[:, 1].astype('int32'), color_scheme[:, 2].astype('int32')] = count_scheme
        
    np_i = rgb_values.flatten().astype('int32')
    #print(np_i.shape, np_i)
    color_scheme_distances = Distances(np_i, df)

    print(type(color_scheme_distances), "\n","\n",color_scheme_distances)
    smallest_list = []
    for path in color_scheme_distances["path"]:
        smallest_list.append('\\'.join(path.split("\\")[:-1]))
        smallest_list.append(path.split("\\")[-1])
    
    return smallest_list




# start = datetime.datetime.now()
# print("\nStart:", start, ", time:")

# print("Start")
# #Full_Preperation()
# Full_Prediction()
# print("end")

# end = datetime.datetime.now()
# duration = end - start

# print("\nStart:", start, ", end:", end, ", duration:", duration)

# mycursor.close()
# mydb.close()

# speichern als lost(filename,pfad) x 5
