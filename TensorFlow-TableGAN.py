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

words = tabletomemory.random_word_dic()

def load(image_file, words):
    image = tf.io.read_file(image_file)
    image = tf.image.decode_jpeg(image)
    
    w = tf.shape(image)[1]