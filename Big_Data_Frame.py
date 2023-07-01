from PIL import Image, ImageFile
import numpy as np
import pandas as pd
from tqdm import tqdm
import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import sqlite3
#from sqlalchemy import create_engine
ImageFile.LOAD_TRUNCATED_IMAGES = True

sqlPath = "../databases/my_database.db"
imagesPath = "D:\\Ablage\\weather_image_recognition"
predictPath = "D:\\Ablage\\predict"

mydb = sqlite3.connect(sqlPath)
mycursor = mydb.cursor()



"""Database"""
# def CreateDatabase():
#     #just execute it if the Database isn't implemented
#     mycursor.execute("create table images(iID int primary key not null, ipath varchar(120) unique);")
#     mydb.commit()
#     mycursor.execute("create table embeddings(eID int, erow int(100), ecol int(100), r int(3), g int(3), b int(3), primary key(eID, erow, ecol), foreign key (eID) references images(iID))")
#     mydb.commit()
#     mycursor.execute("create table schemes(sID int, r int(3), g int(3), b int(3), amount int(10), primary key(sID, r, g, b), foreign key (sID) references images(iID))")
#     mydb.commit()
#     mycursor.execute("create table schemes_distances(ID1 int, ID2 int, Euclidean double, Manhattan double, Cosine double, Jaccard double, Hamming double, primary key(ID1, ID2), foreign key (ID1) references images(iID), foreign key (ID2) references images(iID))")
#     mydb.commit()

def CreateDatabase():
    # just execute it if the Database isn't implemented
    mycursor.execute("CREATE TABLE IF NOT EXISTS images(iID INT PRIMARY KEY NOT NULL, ipath VARCHAR(120) UNIQUE);")
    mydb.commit()
    #mycursor.execute("CREATE TABLE IF NOT EXISTS embeddings(eID INT, erow INT, ecol INT, r INT, g INT, b INT, PRIMARY KEY(eID, erow, ecol), FOREIGN KEY (eID) REFERENCES images(iID));")
    #mydb.commit()
    mycursor.execute("CREATE TABLE IF NOT EXISTS schemes(sID INT, r INT, g INT, b INT, amount INT, PRIMARY KEY(sID, r, g, b), FOREIGN KEY (sID) REFERENCES images(iID));")
    mydb.commit()
    


def CheckDatabase():
    # check if the Database is implemented
    sql_query = f'SELECT name FROM sqlite_master WHERE name="schemes_distances"'
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
    
def ReadPathbyID(id):
    sql_query = f'SELECT ipath FROM images where iID="{id}"'
    result = pd.read_sql(sql_query, con=mydb)
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
    sql_query = f'SELECT r, g, b, amount FROM schemes where sID="{id}"'
    result = pd.read_sql(sql_query, con=mydb)
    if len(result["amount"]) > 0:
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

def writePixels(id,row,col,r,g,b):
    mycursor.execute(f'insert into embeddings values({id}, {row}, {col}, {r}, {g}, {b})')
    mydb.commit()

def writeSchemes(id,r,g,b,amount):
    mycursor.execute(f'insert into schemes values({id}, {r}, {g}, {b}, {amount})')
    mydb.commit()

def writeDistances(id1,id2,e,m,c,j,h):
    #print(f'insert into schemes values({id1}, {id2}, {e}, {m}, {c}, {j}, {h}) ON DUPLICATE KEY UPDATE Euclidean={e}, Manhattan={m}, Cosine={c}, Jaccard={j}, Hamming={h}')
    mycursor.execute(f'insert into schemes_distances values({id1}, {id2}, {e}, {m}, {c}, {j}, {h}) ON DUPLICATE KEY UPDATE Euclidean={e}, Manhattan={m}, Cosine={c}, Jaccard={j}, Hamming={h}')
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
    vectors = np.array(vectors)
    unique_vectors, vector_counts = np.unique(vectors, axis=0, return_counts=True)
    
    converted_vectors = np.where((unique_vectors % 15) >= 3, unique_vectors + (15 - (unique_vectors % 15)), unique_vectors - (unique_vectors % 15))
    
    unique_converted, converted_counts = np.unique(converted_vectors, axis=0, return_counts=True)
    
    return unique_converted, converted_counts

def GoogleAPI():
    pass

def Image_to_rgb_scheme(image):
    # Convert the image to RGB mode if it's not already in RGB mode
    image = image.convert("RGB")
    
    # Get the width and height of the image
    width, height = image.size
    im3d = np.array(image)
    im = im3d.reshape(-1, im3d.shape[-1])
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
            #print(image_id, x, y, r, g, b)
            #writePixels(image_id, x, y, r, g, b)

def PreparationCNN():
    pass

def CNN():
    pass

def Full_Preperation():
    # Loop through each file in the directory
    image_generator = Path_generator(imagesPath)

    maxID = imageMaxID()
    counter = 0

    if maxID is None:
        print("all good")
    else:
        counter = int(maxID)

    # Open the image file
    for image_path in image_generator:


        image_id = ReadIDbyPath(image_path)
        if image_id is None:
            counter += 1

            writeImage(counter, image_path)
            try:
                image = Image.open(image_path)
                vectors = Image_to_rgb_scheme(image)

                color_scheme, count_scheme = Get_color_scheme(vectors)
                
                df = pd.DataFrame({
                    'sID': counter,
                    'r': [cols[0] for cols in color_scheme],
                    'g': [cols[1] for cols in color_scheme],
                    'b': [cols[2] for cols in color_scheme],
                    'amount': count_scheme
                })
                print(counter, color_scheme.shape, len(count_scheme))
                
                #print("\n", [cols for cols in count_scheme])

                # Insert the data into the database
                df.to_sql(name='schemes', con=mydb, if_exists='append',index = False, chunksize = 1000)
            except:
                print("continue")
            #print(color_scheme, "\n", count_scheme)
        #print(len(vectors))
        #image = image.resize((400,400))
        #Image_to_rgb_image(image, image_id)
        #print(400*400)

def predictschemes_gen(image):
    vectors = Image_to_rgb_scheme(image)

    color_scheme, count_scheme = Get_color_scheme(vectors)
    np_i = np.zeros((18, 18, 18))
    for [r, g, b], a in zip(color_scheme, count_scheme):
        np_i[int(r/15)][int(g/15)][int(b/15)] = int(a)
    np_i = np_i.reshape(-1)
    #print(color_scheme.shape, len(count_scheme))
    ids = ReadALLID()
    for j in ids:
        df_j = ReadSchemesbyID(j)
        if df_j is not None:
            np_j = np.zeros((18, 18, 18))
            for x , [r, g, b, a] in df_j.iterrows():
                np_j[int(r/15)][int(g/15)][int(b/15)] = int(a)
            #print(i,j)
            np_j = np_j.reshape(-1)
            e = Distances(j, np_i,np_j)
            yield j, e 

    
"""Code"""
def Input():
    pass

def Distances(id, la,lb):
    a = np.array(la)
    b = np.array(lb)

    euclidean = np.linalg.norm(a - b)
    
    return euclidean


def KNN_color_scheme():
    pass

def KNN_API():
    pass

def Full_Prediction():
    
    data = pd.DataFrame(columns=("name", "place", "e", "ec"))
    image_generator = Path_generator(predictPath)
    for image_path in image_generator:
        image_name = os.path.split(image_path)[-1]

        image = Image.open(image_path)
        img_plot = plt.imshow(image)
        plt.axis('off')  # Turn off the axis
        plt.pause(1)
        schemes = predictschemes_gen(image)
        #predictembeddings_gen(image)
        #predictlabels_gen(image)
        endlist = []
        valmin = 1000000
        for id, val in schemes:
            endlist.append(val)
            if val < valmin:
                newpath = ReadPathbyID(id)
                new_image = mpimg.imread(newpath)
                img_plot.set_data(new_image)
                plt.show(block=False)
                #plt.draw()
                plt.pause(0.05)
        #  IDs und Werte der fünf kleinsten e-Werte
        smallest_e = sorted(enumerate(endlist), key=lambda x: x[1])[:5]
        e_data = pd.DataFrame([(image_name, place, value, int(idx)) for place, (idx, value) in enumerate(smallest_e, 1)], columns=["name", "place", "e", "ec"])
        data = pd.concat([data, e_data], ignore_index=True)
        print(e_data["ec"])

        fig, axes = plt.subplots(1, 6, figsize=(15, 3))
        
        endimage = Image.open(image_path)
        axes[0].imshow(endimage)
        axes[0].axis('off')
        for i, endid in enumerate(e_data["ec"]):
            print(endid)
            image_path = ReadPathbyID(endid)
            print(image_path)
            endimage = Image.open(image_path)
            axes[i+1].imshow(endimage)
            axes[i+1].axis('off')

        # Adjust the spacing between subplots
        plt.tight_layout()

        # Display the plot
        plt.show()


def Output():
    pass



print("Start")
Full_Preperation()
#Full_Prediction()
print("end")
mycursor.close()
mydb.close()