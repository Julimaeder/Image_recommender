# -*- coding: utf-8 -*-
#from glob import glob
import os
import random
from tqdm import tqdm
#%%

path = 'C:\\Users\\maede\\Desktop\\Big_Data\\train\\'

def rename():
    for i, filename in tqdm(enumerate(os.listdir(path))):
        if random.randint(0,10) >= 2:
            os.rename(path + filename, path + str(i) + "_train.jpg")
        else:
            os.rename(path + filename, path + str(i) + "_test.jpg")

rename()

#%%
def rename_back():
    for i, filename in tqdm(enumerate(os.listdir(path))):
            os.rename(path + filename, path + str(i) + ".jpg")


rename_back()

#%%
def generator_filenames():
    num = -1
    while True:
        num += 1
        name = f"{num}_train.jpg"
        yield name
#%%
my_gen = generator_filenames()

#%%
def startfile():
    i = 0
    for i in range(10):
        while True:
            try:
                os.startfile(path+next(my_gen))
            except:
                continue
            break
        i += 1
startfile()
