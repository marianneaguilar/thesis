
import os
os.chdir("/Users/marianneaguilar/Documents")
os.getcwd()
import csv
import pip
import subprocess
import sys
def install(package):
    subprocess.call([sys.executable, "-m", "pip", "install", package])
install("numpy")
import numpy


#Transfer data over
data = []
with open("MIT-college-sleep-diary-20180919-mta.csv") as csvfile:
    reader = csv.reader(csvfile)
    for row in reader: 
        data.append(row)
column_names = data[0]
data = numpy.delete(data, (0), axis=0)

#Visualization
print(data[0])
print(data[0][9])

install("matplotlib")
import matplotlib
install("scipy")
import scipy
import matplotlib.pyplot as plt
install("pandas")
import pandas as pd

#Visualize day-to-day plot before standardization
individuals = numpy.unique(data[:,9])
plt.axes()
def make_num(column):
    x=[]
    for entry in data[:,[column]]:
        try:
            x.append(float(entry))
        except:
            x.append(numpy.nan)
    return x
x=make_num(2)
y=make_num(56)
s=make_num(20)
plt.scatter(x[0:29],y[0:29],c=s[0:29])
plt.xlim(0,31)
plt.ylim(0,100)
plt.xlabel("Day")
plt.ylabel("Sadness")
plt.title("Sadness over time as sleep varied")
plt.show()

#PART 1: DESCRIPTIVE STATISTICS
def histo(column):
    temp=[]
    for entry in data[:,column]:
        try:
            temp.append(float(entry))
        except:
            temp.append(numpy.nan)
    plt.hist(temp,range=[0,100])
    plt.xlabel(column_names[column])
    plt.ylabel("Frequency")
    plt.title("Histogram of "+column_names[column])
    plt.show()
histo(55)
histo(56)
histo(57)
histo(58)
histo(59)

#heat map
import plotly.plotly as py
import plotly.graph_objs as go
install("seaborn")
import seaborn as sns

def make_heat(column):
    st=make_num(column)
    t = pd.DataFrame({"A": data[:,9],
                    "B": x,
                    "C": st})
    table = pd.pivot_table(t,values='C',index=['A'],columns=['B'])
    ax = sns.heatmap(table)
    plt.title(column_names[column]+" over time per person")
    plt.show()

def make_cluster(column):
    st=make_num(column)
    t = pd.DataFrame({"A": data[:,9],
                    "B": x,
                    "C": st})
    table = pd.pivot_table(t,values='C',index=['A'],columns=['B'])
    table = table.fillna(-1)
    ax = sns.clustermap(table.iloc[:,0:30],col_cluster=False)
    plt.title(column_names[column]+" over time per person")
    plt.show()
    
#of_interest = [4,5,6,7,8,15,19,20,55,56,57,58,59,77,123,124,125,126,127]
#for i in of_interest:
 #   make_heat(i)
  #  make_cluster(i)


#Proposed: cluster starting with everything together. Then separate into two groups such that we find two groups with minimum distance between them.
#Con: O(n^2*nCx)
#Alt.:find linear function of variables of interest and cluster as before using this
