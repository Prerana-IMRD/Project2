import sys
print("Importing matplotlib...")
import matplotlib
matplotlib.use('Agg')
print("Importing pandas...")
import pandas as pd
print("Importing numpy...")
import numpy as np
print("Importing matplotlib.pyplot...")
import matplotlib.pyplot as plt
print("Importing seaborn...")
import seaborn as sns
print("Importing gradio...")
import gradio as gr
print("Importing LabelEncoder...")
from sklearn.preprocessing import LabelEncoder
print("Importing f_classif...")
from sklearn.feature_selection import f_classif
print("Importing ExtraTreesClassifier...")
from sklearn.ensemble import ExtraTreesClassifier
print("Importing GaussianNB...")
from sklearn.naive_bayes import GaussianNB
print("Importing DecisionTreeClassifier...")
from sklearn.tree import DecisionTreeClassifier
print("Importing KNeighborsClassifier...")
from sklearn.neighbors import KNeighborsClassifier
print("Importing MLPClassifier...")
from sklearn.neural_network import MLPClassifier
print("Importing StratifiedKFold...")
from sklearn.model_selection import StratifiedKFold
print("Importing accuracy_score...")
from sklearn.metrics import accuracy_score
print("All imports successful!")
