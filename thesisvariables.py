
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
data2_per=[]
for row in data2:
    data2_per.append(scipy.signal.periodogram(row)[1])
data3=[]
person=0
while person<4522:
    temp=[]
    for i in range(0,19):
        temp2=numpy.array(data2[person+i])
        if len(temp2) != 29:
            temp3=numpy.pad(temp2,(0,29-len(temp2)),'constant',constant_values=(0,0))
            temp.append(temp3)
        else:
            temp.append(temp2)
    temp=numpy.array(temp)
    data3.append(temp.flatten())
    person=person+19

install("sklearn")
from sklearn.cluster import FeatureAgglomeration
agglo=FeatureAgglomeration(n_clusters=4,affinity="euclidean")
X=numpy.array(data3)
X=numpy.nan_to_num(X,copy=True)
Z=agglo.fit(numpy.transpose(X))
Z.labels_


def make_grouped_clusters(of_int):
    person=0
    group0=[]
    group1=[]
    group2=[]
    group3=[]
    while person < 238:
        if Z.labels_[person] == 0:
            group0.append(data2[19*person+of_int])
        if Z.labels_[person] == 1:
            group1.append(data2[19*person+of_int])
        if Z.labels_[person] == 2:
            group2.append(data2[19*person+of_int])
        if Z.labels_[person] == 3:
            group3.append(data2[19*person+of_int])
        person=person+1
    t = pd.DataFrame(group0)
    ax = sns.heatmap(t,cbar=False)
    plt.title("Non-PerCluster 0-"+str(of_int))
    plt.savefig('nonpercluster0'+str(of_int)+'.png')
    t1 = pd.DataFrame(group1)
    ax1 = sns.heatmap(t1,cbar=False)
    plt.title("Non-perCluster 1-"+str(of_int))
    plt.savefig('nonpercluster1'+str(of_int)+'.png')
    t2 = pd.DataFrame(group2)
    ax2 = sns.heatmap(t2,cbar=False)
    plt.title("Non-perCluster 2-"+str(of_int))
    plt.savefig('nonpercluster2'+str(of_int)+'.png')
    t3 = pd.DataFrame(group3)
    ax3 = sns.heatmap(t3,cbar=False)
    plt.title("Non-perCluster 3-"+str(of_int))
    plt.savefig('nonpercluster3'+str(of_int)+'.png')
    return group0,group1,group2,group3

sigdiff=[]
for i in range(0,19):
    group0,group1,group2,group3=make_grouped_clusters(i)

"""    
for i in range(0,19):
    group0,group1,group2,group3=make_grouped_clusters(i)
    m1=0
    m2=0
    m3=0
    m4=0
    c1=0
    c2=0
    c3=0
    c4=0
    for row in numpy.array(group0):
        m1=m1+sum(row)
        c1=c1+len(row)
    for row in numpy.array(group1):
        m2=m2+sum(row)
        c2=c2+len(row)
    for row in numpy.array(group2):
        m3=m3+sum(row)
        c3=c3+len(row)
    for row in numpy.array(group3):
        m4=m4+sum(row)
        c4=c4+len(row)
    if c1 == 0:
        avg1=0
    else:
        avg1=m1/c1
    if c2 == 0:
        avg2=0
    else:
        avg2=m2/c2
    if c3 == 0:
        avg3=0
    else:
        avg3=m3/c3
    if c4 == 0:
        avg4=0
    else:
        avg4=m4/c4 
    sigdiff.append([avg1,avg2,avg3,avg4])



sigdiffstd=[]

sigdiff2=[]
for row in sigdiff:
    group0=row[0]
    group1=row[1]
    group2=row[2]
    group3=row[3]
    sigdiff2.append(scipy.stats.ttest_ind(group0,group1))
    sigdiff2.append(scipy.stats.ttest_ind(group0,group2))
    sigdiff2.append(scipy.stats.ttest_ind(group0,group3))
    sigdiff2.append(scipy.stats.ttest_ind(group2,group1))
    sigdiff2.append(scipy.stats.ttest_ind(group3,group1))
    sigdiff2.append(scipy.stats.ttest_ind(group2,group3))
    
#Analyze 3D relationships
#import thesis3d

#Standardize
#import stand


    
