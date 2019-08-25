# -*- coding: utf-8 -*-
"""
Created on Sun Jul 28 14:25:24 2019

@author: Nick

"""

#Code taken from:
#https://stackoverflow.com/questions/19726663/how-to-save-the-pandas-dataframe-series-data-as-a-figure/39358752#39358752
#https://stackoverflow.com/questions/43564943/saving-matplotlib-plot-to-memory-and-placing-on-tkinter-canvas

#to mess around with the image:
#https://stackoverflow.com/questions/11837979/removing-white-space-around-a-saved-image-in-matplotlib/27227718

#For the output, copy this:
#https://github.com/tensorflow/tensorflow/blob/r1.13/tensorflow/contrib/eager/python/examples/generative_examples/image_captioning_with_attention.ipynb


#For tokenization of dictionary:
#https://www.tensorflow.org/beta/tutorials/text/image_captioning

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import six
import io
from PIL import Image
from random import randrange
import urllib.request
import random

import tensorflow as tf


def generate_word_list():
    word_url = "http://svnweb.freebsd.org/csrg/share/dict/words?view=co&content-type=text/plain"
    response = urllib.request.urlopen(word_url)
    long_txt = response.read().decode()
    words = long_txt.splitlines()
    return words

#function to render the data
def render_mpl_table(data, col_width=3.0, row_height=0.625, font_size=14,
                     header_color='#40466e', row_colors=['#f1f1f2', 'w'], edge_color='w',
                     bbox=[0,0,1,1], header_columns=0,
                     ax=None, **kwargs):
    if ax is None:
        size = (np.array(data.shape[::-1]) + np.array([0, 1])) * np.array([col_width, row_height])
        fig, ax = plt.subplots(figsize=(8,5))
        ax.axis('off')

    mpl_table = ax.table(cellText=data.values, bbox=bbox, colLabels=data.columns, **kwargs)

    mpl_table.auto_set_font_size(True)
    #mpl_table.set_fontsize(font_size)

    for k, cell in six.iteritems(mpl_table._cells):
        cell.set_edgecolor(edge_color)
        if k[0] == 0 or k[1] < header_columns:
            cell.set_text_props(weight='bold', color='w')
            cell.set_facecolor(header_color)
        else:
            cell.set_facecolor(row_colors[k[0]%len(row_colors) ])
    return ax,fig

#Example data
words = generate_word_list()


class random_word_dic:
    def __init__(self,):
        self.dic = generate_word_list() 


rows = random.randint(1,4)
columns = random.randint(1,4)
testframe = pd.DataFrame()
numberframe = pd.DataFrame()

for const in random.sample(words, columns):
    print(const)
    #x = random.randint(0, 1)
    #if x == 1:
    row = random.sample(words,rows)
    ind_row = [words.index(s) for s in row]
    testframe[const] = row
    numberframe[words.index(const)] = ind_row
    #else:
        #testframe[const] = random.sample(range(1, 5000), rows)


#Utility function to pad the lists
def pad(l, content, width):
    l.extend([content] * (width - len(l)))
    return l

#get header



#get number of rows and columns
r,c = testframe.shape
final_data = np.zeros((5,4))
final_data[0,:] = pad(list(numberframe),0,4)
for r in range(testframe.shape[0]):
    print(r)
    final_data[r+1,:] = pad(list(numberframe.iloc[r]),0,4)

#df = pd.DataFrame()
#df['date'] = ['2016-04-01', '2016-04-02']
#df['calories'] = [2200, 2100]
#df['sleep hours'] = [2200, 2100]
#df['gym'] = [True, False]
#df['weights'] = ['BIGemall', 'SMALL']



#outputs a ? + figure

#[a,b] = render_mpl_table(testframe, header_columns=0, col_width=2.0)

#width = 100*3
#height = 100

#save figure memory as IM

#buf = io.BytesIO()
#b.savefig(buf,format='jpeg',pad_inches=0,bbox_inches = 'tight')
#buf.seek(0)
#m = Image.open(buf)

#*****
#imgplot = plt.imshow(im)
#plt.show()