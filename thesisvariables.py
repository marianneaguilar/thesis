
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
    
of_interest = [4,5,6,7,8,15,19,20,55,56,57,58,59,77,123,124,125,126,127]
for i in of_interest:
    make_heat(i)
    make_cluster(i)


#STANDARDIZE
data_numeric = []
    
for column in of_interest:
    temp=make_num(column)
    temp2 = pd.DataFrame({"A" : data[:,9],"B": temp})
    temp3 = pd.pivot_table(temp2, values='B',index=['A'],aggfunc=numpy.mean)
    temp4 = pd.pivot_table(temp2, values='B',index=['A'],aggfunc=numpy.std)
    for i in range(0,7200):
        person = data[i,9]
        sd = temp4.query('A == @person')
        mn = temp3.query('A == @person')
        if sd.B[0] > 0:
            temp[i]=(temp[i]-mn.B[0])/sd.B[0]
        else:
            temp[i]=numpy.nan
    data_numeric.append(temp)

data_numeric2 = [list(i) for i in zip(*data_numeric)]
data_numeric3 = pd.DataFrame(data_numeric2,columns=of_interest)
corrs=data_numeric3.corr()

list_names=[]
for i in of_interest:
    list_names.append(column_names[i])
ax2 = sns.heatmap(corrs,xticklabels=list_names,yticklabels=list_names)
plt.show()

#Analyze 3D relationships
from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure()
ax3 = Axes3D(fig)
ax3.plot(xs=x,ys=make_num(56),zs=make_num(20))
ax3.set_xlabel('Day of month')
ax3.set_ylabel('Mood')
ax3.set_zlabel('Sleep time')
plt.show()

#Analyze individual students in 3D plot

install("colour")
from colour import Color
def analyze_relationships_3d(col1, col2):
    red = Color("red")
    colors = list(red.range_to(Color("violet"),len(indivs)+1))
    indivs2 = list(set(data[:,9]))
    for pos in range(0,(len(indivs2)-1)):
        indiv=indivs2[pos]
        temp_xs=data[data[:,9]==indiv,2]
        temp_mood=data[data[:,9]==indiv,col1]
        temp_sleep=data[data[:,9]==indiv,col2]
        xse=[]
        for entry in temp_xs:
            try:
                xse.append(float(entry))
            except:
                xse.append(numpy.nan)
        mood=[]
        for entry in temp_mood:
            try:
                mood.append(float(entry))
            except:
                mood.append(numpy.nan)
        sleep=[]
        for entry in temp_sleep:
            try:
                sleep.append(float(entry))
            except:
                sleep.append(numpy.nan)
        ax4.plot(xs=xse,ys=mood,zs=sleep,c=colors[pos+1].hex)

fig=plt.figure()
ax4 = Axes3D(fig)
analyze_relationships_3d(56,20)        
ax4.set_xlabel('Day of month')
ax4.set_ylabel('Mood')
ax4.set_zlabel('Sleep time')
plt.show()

