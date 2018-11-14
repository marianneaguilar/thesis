
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
"""
print(data[0])
print(data[0][9])
"""

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

#Removed 17,55,57,58,59,60,61,62,63,64,65,66,125,126,127,128,129,130,131,132,133,134
of_interest = [4,5,6,7,14,15,18,19,20,21,22,23,24,25,79,111,112,113,114,115,117,118,136]
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
def fix(of_interest):
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
    while person<238*len(of_interest):
        temp=[]
        for i in range(0,len(of_interest)):
            temp2=numpy.array(data2_per[person+i])
            if len(temp2) != 29:
                temp3=numpy.pad(temp2,(0,29-len(temp2)),'constant',constant_values=(0,0))
                temp.append(temp3)
            else:
                temp.append(temp2)
        temp=numpy.array(temp)
        data3.append(temp.flatten())
        person=person+len(of_interest)
    return data2,data3

data2,data3=fix(of_interest)

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
            group0.append(data2[len(of_interest)*person+of_int])
        if Z.labels_[person] == 1:
            group1.append(data2[len(of_interest)*person+of_int])
        if Z.labels_[person] == 2:
            group2.append(data2[len(of_interest)*person+of_int])
        if Z.labels_[person] == 3:
            group3.append(data2[len(of_interest)*person+of_int])
        person=person+1
        """
    t = pd.DataFrame(group0)
    ax = sns.heatmap(t,cbar=False)
    plt.title("PerCluster 0-"+str(of_int))
    plt.savefig('percluster0'+str(of_int)+'.png')
    t1 = pd.DataFrame(group1)
    ax1 = sns.heatmap(t1,cbar=False)
    plt.title("PerCluster 1-"+str(of_int))
    plt.savefig('percluster1'+str(of_int)+'.png')
    t2 = pd.DataFrame(group2)
    ax2 = sns.heatmap(t2,cbar=False)
    plt.title("PerCluster 2-"+str(of_int))
    plt.savefig('percluster2'+str(of_int)+'.png')
    t3 = pd.DataFrame(group3)
    ax3 = sns.heatmap(t3,cbar=False)
    plt.title("PerCluster 3-"+str(of_int))
    plt.savefig('percluster3'+str(of_int)+'.png')
    """
    return group0,group1,group2,group3

"""
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
    """

sigdiff=[]
comparison=[]
    
for i in range(0,len(of_interest)):
    group0,group1,group2,group3=make_grouped_clusters(i)
    m1=[]
    m2=[]
    m3=[]
    m4=[]
    for row in numpy.array(group0):
        m1.append(numpy.mean(row))
    for row in numpy.array(group1):
        m2.append(numpy.mean(row))
    for row in numpy.array(group2):
        m3.append(numpy.mean(row))
    for row in numpy.array(group3):
        m4.append(numpy.mean(row)) 
    sigdiff.append(scipy.stats.ttest_ind(m1,m2,equal_var=False))
    comparison.append("Group 1-Group 2 on "+column_names[of_interest[i]])
    sigdiff.append(scipy.stats.ttest_ind(m1,m3,equal_var=False))
    comparison.append("Group 1-Group 3 on "+column_names[of_interest[i]])
    sigdiff.append(scipy.stats.ttest_ind(m1,m4,equal_var=False))
    comparison.append("Group 1-Group 4 on "+column_names[of_interest[i]])
    sigdiff.append(scipy.stats.ttest_ind(m3,m2,equal_var=False))
    comparison.append("Group 2-Group 3 on "+column_names[of_interest[i]])
    sigdiff.append(scipy.stats.ttest_ind(m4,m2,equal_var=False))
    comparison.append("Group 2-Group 4 on "+column_names[of_interest[i]])
    sigdiff.append(scipy.stats.ttest_ind(m3,m4,equal_var=False))
    comparison.append("Group 3-Group 4 on "+column_names[of_interest[i]])


for i in range(0,len(sigdiff)):
    if sigdiff[i].pvalue<0.1:
        print(comparison[i])

#To input all numeric data
of_interest2 = [4,5,6,7,14,15,17,18,19,20,21,22,23,24,25,55,57,58,59,60,61,62,63,64,65,66,79,111,112,113,114,115,117,118,125,126,127,128,129,130,131,132,133,134,136]

data2,data3=fix(of_interest2)

group0=[]
group1=[]
group2=[]
group3=[]
group0ans=[]
group0indivs=[]
group1ans=[]
group1indivs=[]
group2ans=[]
group2indivs=[]
group3ans=[]
group3indivs=[]
temp=data[data[:,2]=='28',:]

#Create answers to what mood[day 28] is
for row in range(0,len(Z.labels_)):
    t=temp[temp[:,9]==individuals[row],:]
    if Z.labels_[row]==0:
        if len(t) == 1:
            group0ans.append(t[0,[57,58,60]])
            group0.append(data3[row])
            group0indivs.append(individuals[row])
    if Z.labels_[row]==1:
        if len(t) == 1:
            group1ans.append(t[0,[57,58,60]])
            group1.append(data3[row])
            group1indivs.append(individuals[row])
    if Z.labels_[row]==2:
        if len(t) == 1:
            group2ans.append(t[0,[57,58,60]])
            group2.append(data3[row])
            group2indivs.append(individuals[row])
    if Z.labels_[row]==3:
        if len(t) == 1:
            group3ans.append(t[0,[57,58,60]])
            group3.append(data3[row])
            group3indivs.append(individuals[row])

#Create neural network        
from sklearn.neural_network import MLPRegressor
neural=MLPRegressor(activation='identity',solver='lbfgs')

lim0=int(numpy.trunc(len(group0)/2))
lim1=int(numpy.trunc(len(group1)/2))
lim2=int(numpy.trunc(len(group2)/2))
lim3=int(numpy.trunc(len(group3)/2))

#Separate into training and testing
group0tester=group0[0:lim0]
group0testerans=group0ans[0:lim0]
group0fit=group0[1+lim0:len(group0)]
group0fitans=group0ans[1+lim0:len(group0ans)]
group1tester=group1[0:lim1]
group1testerans=group1ans[0:lim1]
group1fit=group1[1+lim1:len(group1)]
group1fitans=group1ans[1+lim1:len(group1ans)]
group2tester=group2[0:lim2]
group2testerans=group2ans[0:lim2]
group2fit=group2[1+lim2:len(group2)]
group2fitans=group2ans[1+lim2:len(group2ans)]
group3tester=group3[0:lim3]
group3testerans=group3ans[0:lim3]
group3fit=group3[1+lim3:len(group3)]
group3fitans=group3ans[1+lim3:len(group3ans)]

#Train model for group 0
X0=numpy.array(group0tester)
X0=numpy.nan_to_num(X0,copy=True)
X1=numpy.array(group0testerans)
X1=numpy.nan_to_num(X1,copy=True)
group0model=neural.fit(X0,X1)

#Train model for group 1
X2=numpy.array(group1tester)
X2=numpy.nan_to_num(X2,copy=True)
X3=numpy.array(group1testerans)
X3=numpy.nan_to_num(X3,copy=True)
group1model=neural.fit(X2,X3)

#Train model for group 2
X4=numpy.array(group2tester)
X4=numpy.nan_to_num(X4,copy=True)
X5=numpy.array(group2testerans)
X5=numpy.nan_to_num(X5,copy=True)
group2model=neural.fit(X4,X5)

#Train model for group 3
X6=numpy.array(group3tester)
X6=numpy.nan_to_num(X6,copy=True)
X7=numpy.array(group3testerans)
X7=numpy.nan_to_num(X7,copy=True)
group3model=neural.fit(X6,X7)

#Test model for group 0
X8=numpy.array(group0fit)
X8=numpy.nan_to_num(X8,copy=True)
group0preds=group0model.predict(X8)

#Test model for group 1
X9=numpy.array(group1fit)
X9=numpy.nan_to_num(X9,copy=True)
group1preds=group1model.predict(X9)

#Test model for group 2
X10=numpy.array(group2fit)
X10=numpy.nan_to_num(X10,copy=True)
group2preds=group2model.predict(X10)

#Test model for group 3
X11=numpy.array(group3fit)
X11=numpy.nan_to_num(X11,copy=True)
group3preds=group3model.predict(X11)


#Analyze 3D relationships
#import thesis3d

#Standardize
#import stand
