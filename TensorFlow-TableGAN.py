# -*- coding: utf-8 -*-
"""
Created on Sun Jul 28 20:09:25 2019

@author: Nick
"""
#https://www.tensorflow.org/beta/tutorials/generative/pix2pix

from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf

import os
import time
import matplotlib.pyplot as plt
from IPython.display import clear_output
from random import randrange
import tabletomemory
import random
import pandas as pd
import numpy as np
import io
from PIL import Image

words = tabletomemory.random_word_dic()

#Instead of loading a file, we will call this generate - generate a training example and output table
def generate(words):
    
    #Utility function to pad the lists
    def pad(l, content, width):
        l.extend([content] * (width - len(l)))
        return l
    
    #Generate dataframe for table, and random number of rows and columns
    rows = random.randint(1,4)
    columns = random.randint(1,4)
    testframe = pd.DataFrame()
    numberframe = pd.DataFrame()
    
    for const in random.sample(words.dic, columns):
        row = random.sample(words.dic,rows)
        ind_row = [words.dic.index(s) for s in row]
        testframe[const] = row
        numberframe[words.dic.index(const)] = ind_row
    
    #Generate the output numpy array w/ padding
    r,c = testframe.shape
    final_data = np.zeros((5,4))
    final_data[0,:] = pad(list(numberframe),0,4)
    for r in range(testframe.shape[0]):
        final_data[r+1,:] = pad(list(numberframe.iloc[r]),0,4)
    
    #Generate the table image
    [a,b] = tabletomemory.render_mpl_table(testframe, header_columns=0, col_width=2.0)
    
    #Read the table image from memory and store it in tf???
    buf = io.BytesIO()
    b.savefig(buf,format='jpeg',pad_inches=0,bbox_inches = 'tight')
    buf.seek(0)
    
    #image = tf.read_file(b)
    #image = tf.image.decode_jpeg(b)
    
    input_image = Image.open(buf)
    
    return input_image, final_data
