
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
install("spicy")


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
#plt.scatter(x[0:29],y[0:29],c=s[0:29])
#plt.xlim(0,31)
#plt.ylim(0,100)
#plt.xlabel("Day")
#plt.ylabel("Sadness")
#plt.title("Sadness over time as sleep varied")
#plt.show()

#PART 1: DESCRIPTIVE STATISTICS
def histo(column):
    plt.hist(make_num(column),range=[0,100])
    plt.xlabel(column_names[column])
    plt.ylabel("Frequency")
    plt.title("Histogram of "+column_names[column])
    plt.show()
#for i in range(55,59):
 #   histo(i)

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
#for i in of_interest:
 #   make_heat(i)
  #  make_cluster(i)

#Cluster all
import scipy
install("statistics")
import statistics
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from scipy.signal import periodogram,csd
install("math")
import math
#N=number time series objects
#T=length of time serie objects (using 30 days here)
N=len(individuals)
T=30
P=N
#Reorganize data into time series format without using pivot function
data1=[]
data2=[]
for i in individuals:
    for col in of_interest:
        data_temp=data[i==data[:,9],col]
        data_temp2=[]
        for entry in range(0,len(data_temp)-1):
            if data_temp[entry] == "":
                a=0
                c=0
                if entry-14 >= 0:
                    try:
                        a=a+data_temp[entry-14]
                        c=c+1
                    except:
                        a=a+0
                        c=c+0
                if entry-7 >= 0:
                    try:
                        a=a+data_temp[entry-7]
                        c=c+1
                    except:
                        a=a+0
                        c=c+0
                if entry+7 < len(data_temp):
                    try:
                        a=a+data_temp[entry+7]
                        c=c+1
                    except:
                        a=a+0
                        c=c+0
                if entry+14 < len(data_temp):
                    try:
                        a=a+data_temp[entry+14]
                        c=c+1
                    except:
                        a=a+0
                        c=c+0
                if c==0:
                    data_temp2.append(-1)
                else:
                    data_temp2.append(a/c)
            else:
                try:
                    data_temp2.append(float(data_temp[entry]))
                except:
                    a=0
                    c=0
                    if entry-14 >= 0:
                        try:
                            a=a+data_temp[entry-14]
                            c=c+1
                        except:
                            a=a+0
                            c=c+0
                    if entry-7 >= 0:
                        try:
                            a=a+data_temp[entry-7]
                            c=c+1
                        except:
                            a=a+0
                            c=c+0
                    if entry+7 < len(data_temp):
                        try:
                            a=a+data_temp[entry+7]
                            c=c+1
                        except:
                            a=a+0
                            c=c+0
                    if entry+14 < len(data_temp):
                        try:
                            a=a+data_temp[entry+14]
                            c=c+1
                        except:
                            a=a+0
                            c=c+0
                    if c==0:
                        data_temp2.append(-1)
                    else:
                        data_temp2.append(a/c)
        data1.append(data_temp2[0:29])
for row in data1:
    avg=numpy.mean(row)
    s=numpy.std(row)
    data_temp3=[]
    for i in row:
        data_temp3.append((i-avg)/s)
    data2.append(data_temp3)
#Calculate periodogram-cross periodogram matrices
data2_per = []
pos=0
while pos < N*len(of_interest):
    for i in range(0,len(of_interest)):
        for j in range(0,len(of_interest)):
            temp_data2=data2[pos+i]
            temp_data3=data2[pos+j]
            data2_per.append(csd(temp_data2,temp_data3)[1])
    pos=pos+len(of_interest)

t2=len(data2_per)
p2=len(data2_per[1])

#Smooth periodograms and crossperiodograms
m=1
data2_per_sm=[]
for student in data2_per:
    temp_sm=[]
    for col in range(0,p2):
        if col < m:
            temp_sm.append(numpy.mean(student[0:col+m]))
        elif col > (p2-m-1):
            temp_sm.append(numpy.mean(student[col-m:p2]))
        else:
            temp_sm.append(numpy.mean(student[col-m:col+m]))
    data2_per_sm.append(temp_sm)

t3=len(data2_per_sm)
p3=len(data2_per_sm[1])

data2_per_sm2=numpy.nan_to_num(data2_per_sm,copy=True)

t4=len(data2_per_sm2)
p4=len(data2_per_sm2[1])

#Compute likelihood test for each frequency        
Qxyw=[]
comparisons=int(len(of_interest)**2)
data2_per_sm_usable=numpy.transpose(data2_per_sm2)
op=0
for col in range(0,p3):
    for counter in range(0,len(of_interest)):
        temp_col=data2_per_sm_usable[col]
        person1=counter
        while person1 < t3-comparisons:
            person2=person1+comparisons
            while person2 < t3:
                Q_new1=temp_col[person1]
                if math.isnan(Q_new1):
                    Q_new1 = 0
                Q_new2=temp_col[person2]
                if math.isnan(Q_new2):
                    Q_new2 = 0
                Q_new12=Q_new1+Q_new2
                if Q_new12 == 0 or math.isnan(Q_new12):
                    Q_new=(2**(2*(2*m+1)))*(Q_new1**(2*m+1)*Q_new2**(2*m+1))
                else:
                    Q_new=(2**(2*(2*m+1)))*((Q_new1**(2*m+1))*(Q_new2**(2*m+1)))/(Q_new12**(2*(2*m+1)))
                Qxyw.append(Q_new)
                person2=person2+comparisons
            person1=person1+comparisons
Qxy=[]
num_pairs=sum(x for x in range(1,238))*19
pairs=0
while pairs < num_pairs:
    s=0
    for w in range(0,14):
        s=s+Qxyw[pairs+w*num_pairs]
    Qxy.append(abs(s/(15*10**(-50))))
    pairs=pairs+1
Qstar=numpy.reshape(Qxy,(19,28203))
Qstar=numpy.transpose(Qstar)

#Analyze 3D relationships
#import thesis3d

#Standardize
#import stand 
    
