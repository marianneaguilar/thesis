#Import packages needed for installation of other packages
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


#Transfer data from Excel over to numpy array
data = []
with open("MIT-college-sleep-diary-20180919-mta.csv") as csvfile:
    reader = csv.reader(csvfile)
    for row in reader: 
        data.append(row)
column_names = data[0]
data = numpy.delete(data, (0), axis=0)

#Check of successful transfer
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
#Function to make a column numeric and return the numeric values
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
"""
plt.scatter(x[0:29],y[0:29],c=s[0:29])
plt.xlim(0,31)
plt.ylim(0,100)
plt.xlabel("Day")
plt.ylabel("Sadness")
plt.title("Sadness over time as sleep varied")
plt.show()
"""

#PART 1: DESCRIPTIVE STATISTICS
#Function to make histogram
def histo(column):
    plt.hist(make_num(column),range=[0,100])
    plt.xlabel(column_names[column])
    plt.ylabel("Frequency")
    plt.title("Histogram of "+column_names[column])
    plt.show()

"""
for i in range(55,59):
    histo(i)
"""

#Function to make heat map and clustered heat map
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

#PART 2:CLUSTERING ALL THE DATA

#Define columns of interest
#Removed 17,55,57,58,59,60,61,62,63,64,65,66,125,126,127,128,129,130,131,132,133,134
of_interest = [4,5,6,7,14,15,18,19,20,21,22,23,24,25,79,111,112,113,114,115,117,118,136]
"""
for i in of_interest:
    make_heat(i)
    make_cluster(i)
"""

#Import and install packages for clustering
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
#Function to reorganize data into time series format without using pivot function
def fix(of_interest):
    data1=[]
    data2=[]
    for i in individuals: #Per individual
        for col in of_interest: #Per column of interest
            data_temp=data[i==data[:,9],col] #Isolate column of interest
            data_temp2=[]
            for entry in range(0,len(data_temp)-1):
                if data_temp[entry] == "": #Check if empty
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
            data1.append(data_temp2[0:27])
    #Standardize
    for row in data1: 
        avg=numpy.mean(row)
        s=numpy.std(row)
        data_temp3=[]
        for i in row:
            data_temp3.append((i-avg)/s)
        data2.append(data_temp3)
    data2_per=[]
    #Find periodogram
    for row in data2: 
        data2_per.append(scipy.signal.periodogram(row)[1])
    data3=[]
    person=0
    #Flatten so that one person per row of data3
    while person<238*len(of_interest):
        temp=[]
        for i in range(0,len(of_interest)):
            temp2per=numpy.array(data2_per[person+i])
            temp2prestan=numpy.array(data2[person+i])
            temp2stan=[]
            if len(temp2prestan)<27:
                temp2stan=numpy.pad(temp2prestan,27-len(temp2prestan),'constant',constant_values=(0,0))
            else:
                temp2stan=temp2prestan[0:26]
            temp2=numpy.array(numpy.append(temp2per,temp2stan[len(temp2stan)-(35-len(temp2per)):len(temp2stan)]))
            if len(temp2) != 35:
                temp3=numpy.pad(temp2,(0,35-len(temp2)),'constant',constant_values=(0,0))
                temp.append(temp3)
            else:
                temp.append(temp2)
        temp=numpy.array(temp)
        data3.append(temp.flatten())
        person=person+len(of_interest)
    
    return data1,data2,data3

data1,data2,data3=fix(of_interest)

#Use sklearn to cluster on features of individuals
install("sklearn")
from sklearn.cluster import FeatureAgglomeration
acti='tanh'
nclus=4
agglo=FeatureAgglomeration(n_clusters=nclus,affinity="euclidean")
X=numpy.array(data3)
X=numpy.nan_to_num(X,copy=True)
Z=agglo.fit(numpy.transpose(X))
Z.labels_

#Function to visualize different heat maps
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

#Function to test if means of individuals' time series across columns differ significantly
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
    if len(m1)>0 and len(m2)>0:
        sigdiff.append(scipy.stats.ttest_ind(m1,m2,equal_var=False))
        comparison.append("Group 1-Group 2 on "+column_names[of_interest[i]])
    if len(m1)>0 and len(m3)>0:
        sigdiff.append(scipy.stats.ttest_ind(m1,m3,equal_var=False))
        comparison.append("Group 1-Group 3 on "+column_names[of_interest[i]])
    if len(m1)>0 and len(m4)>0:
        sigdiff.append(scipy.stats.ttest_ind(m1,m4,equal_var=False))
        comparison.append("Group 1-Group 4 on "+column_names[of_interest[i]])
    if len(m2)>0 and len(m3)>0:
        sigdiff.append(scipy.stats.ttest_ind(m3,m2,equal_var=False))
        comparison.append("Group 2-Group 3 on "+column_names[of_interest[i]])
    if len(m2)>0 and len(m4)>0:
        sigdiff.append(scipy.stats.ttest_ind(m4,m2,equal_var=False))
        comparison.append("Group 2-Group 4 on "+column_names[of_interest[i]])
    if len(m3)>0 and len(m4)>0:
        sigdiff.append(scipy.stats.ttest_ind(m3,m4,equal_var=False))
        comparison.append("Group 3-Group 4 on "+column_names[of_interest[i]])

#Visualize which column of interest in data differ significantly in clustering
for i in range(0,len(sigdiff)):
    if sigdiff[i].pvalue<0.1:
        print(comparison[i])

#To input all numeric data
of_interest2 = [4,5,6,7,14,15,17,18,19,20,21,22,23,24,25,55,57,58,59,60,61,62,63,64,65,66,79,111,112,113,114,115,117,118,125,126,127,128,129,130,131,132,133,134,136]

data1,data2,data3=fix(of_interest2)

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
    t0=0
    t1=0
    t2=0
    m0=numpy.mean(data1[row*len(of_interest2)+16])
    m1=numpy.mean(data1[row*len(of_interest2)+17])
    m2=numpy.mean(data1[row*len(of_interest2)+19])
    try:
        t0=float(t[0][57])
    except:
        t0=m0
    try:
        t1=float(t[0][58])
    except:
        t1=m1
    try:
        t2=float(t[0][60])
    except:
        t2=m2
    if Z.labels_[row]==0:
        if len(t) == 1:
            group0ans.append([t0,t1,t2])
            group0.append(data3[row])
            group0indivs.append(individuals[row])
    if Z.labels_[row]==1:
        if len(t) == 1:
            group1ans.append([t0,t1,t2])
            group1.append(data3[row])
            group1indivs.append(individuals[row])
    if Z.labels_[row]==2:
        if len(t) == 1:
            group2ans.append([t0,t1,t2])
            group2.append(data3[row])
            group2indivs.append(individuals[row])
    if Z.labels_[row]==3:
        if len(t) == 1:
            group3ans.append([t0,t1,t2])
            group3.append(data3[row])
            group3indivs.append(individuals[row])

#Create neural network
#Variation 1: hidden_layer_sizes=10
#Variation 2: hidden_layer_sizes=(10,10)
#Variation 3: hidden_layer_sizes=(10,10,10)
#Variation 4: hidden_layer_sizes=(10,10),solver=adam,max_iter=100000
#Variation 5:
from sklearn.neural_network import MLPRegressor
neural0=MLPRegressor(hidden_layer_sizes=(10,10,10),activation=acti,solver='lbfgs',random_state=0)
neural1=MLPRegressor(hidden_layer_sizes=(10,10,10),activation=acti,solver='lbfgs',random_state=0)
neural2=MLPRegressor(hidden_layer_sizes=(10,10,10),activation=acti,solver='lbfgs',random_state=0)
neural3=MLPRegressor(hidden_layer_sizes=(10,10,10),activation=acti,solver='lbfgs',random_state=0)

lim0=int(numpy.trunc(len(group0)/2))
lim1=int(numpy.trunc(len(group1)/2))
lim2=int(numpy.trunc(len(group2)/2))
lim3=int(numpy.trunc(len(group3)/2))

#Separate into training and testing
group0tester=[]
group0testerans=[]
group0fit=[]
group0fitans=[]
group1tester=[]
group1testerans=[]
group1fit=[]
group1fitans=[]
group2tester=[]
group2testerans=[]
group2fit=[]
group2fitans=[]
group3tester=[]
group3testerans=[]
group3fit=[]
group3fitans=[]
if len(group0ans)>0 and len(group0)>0:
    group0tester=group0[0:lim0]
    group0testerans=group0ans[0:lim0]
    group0fit=group0[1+lim0:len(group0)]
    group0fitans=group0ans[1+lim0:len(group0ans)]
if len(group1ans)>0 and len(group1)>0:
    group1tester=group1[0:lim1]
    group1testerans=group1ans[0:lim1]
    group1fit=group1[1+lim1:len(group1)]
    group1fitans=group1ans[1+lim1:len(group1ans)]
if len(group2ans)>0 and len(group2)>0:
    group2tester=group2[0:lim2]
    group2testerans=group2ans[0:lim2]
    group2fit=group2[1+lim2:len(group2)]
    group2fitans=group2ans[1+lim2:len(group2ans)]
if len(group3ans)>0 and len(group3)>0:
    group3tester=group3[0:lim3]
    group3testerans=group3ans[0:lim3]
    group3fit=group3[1+lim3:len(group3)]
    group3fitans=group3ans[1+lim3:len(group3ans)]

#Train model for group 0
X0=[]
X1=[]
X2=[]
X3=[]
X4=[]
X5=[]
X6=[]
X7=[]
X8=[]
X9=[]
X10=[]
X11=[]
if len(group0tester)>0 and len(group0testerans)>0:
    X0=numpy.asarray(group0tester, dtype=float)
    X0=numpy.nan_to_num(X0,copy=True)
    X1=numpy.asarray(group0testerans,dtype=float)
    X1=numpy.nan_to_num(X1,copy=True)
    group0model=neural0.fit(X0,X1)

#Train model for group 1
if len(group1tester)>0 and len(group1testerans)>0:    
    X2=numpy.asarray(group1tester, dtype=float)
    X2=numpy.nan_to_num(X2,copy=True)
    X3=numpy.asarray(group1testerans, dtype=float)
    X3=numpy.nan_to_num(X3,copy=True)
    group1model=neural1.fit(X2,X3)
    

#Train model for group 2
if len(group2tester)>0 and len(group2testerans)>0:  
    X4=numpy.asarray(group2tester, dtype=float)
    X4=numpy.nan_to_num(X4,copy=True)
    X5=numpy.asarray(group2testerans,dtype=float)
    X5=numpy.nan_to_num(X5,copy=True)
    group2model=neural2.fit(X4,X5)

#Train model for group 3
if len(group3tester)>0 and len(group3testerans)>0:  
    X6=numpy.asarray(group3tester,dtype=float)
    X6=numpy.nan_to_num(X6,copy=True)
    X7=numpy.asarray(group3testerans,dtype=float)
    X7=numpy.nan_to_num(X7,copy=True)
    group3model=neural3.fit(X6,X7)

group0preds=[]
group1preds=[]
group2preds=[]
group3preds=[]
#Test model for group 0
if len(group0fit)>0:
    X8=numpy.asarray(group0fit,dtype=float)
    X8=numpy.nan_to_num(X8,copy=True)
    group0preds=group0model.predict(X8)

#Test model for group 1
if len(group1fit)>0:
    X9=numpy.asarray(group1fit,dtype=float)
    X9=numpy.nan_to_num(X9,copy=True)
    group1preds=group1model.predict(X9)

#Test model for group 2
if len(group2fit)>0:
    X10=numpy.asarray(group2fit,dtype=float)
    X10=numpy.nan_to_num(X10,copy=True)
    group2preds=group2model.predict(X10)

#Test model for group 3
if len(group3fit)>0:
    X11=numpy.asarray(group3fit,dtype=float)
    X11=numpy.nan_to_num(X11,copy=True)
    group3preds=group3model.predict(X11)


#Create residual plots and calculate MSE estimate
mse0=0
mse1=0
mse2=0
mse3=0
#Residual plots group 0
resids57_0=[]
for i in range(0,len(group0preds)):
    #print("Prediction: "+str(group0preds[i][0])+" Actual: "+str(group0fitans[i][0]))
    resids57_0.append(group0preds[i][0]-group0fitans[i][0])
    mse0=mse0+(group0preds[i][0]-group0fitans[i][0])**2
"""
plt.hist(resids57_0)
plt.xlabel("Residuals")
plt.ylabel("Frequency")
plt.title("Histogram of Residuals for Scale 1-Model 0")
plt.show()
"""
resids58_0=[]
for i in range(0,len(group0preds)):
    #print("Prediction: "+str(group0preds[i][1])+" Actual: "+str(group0fitans[i][1]))
    resids58_0.append(group0preds[i][1]-group0fitans[i][1])
    mse0=mse0+(group0preds[i][1]-group0fitans[i][1])**2
"""
plt.hist(resids58_0)
plt.xlabel("Residuals")
plt.ylabel("Frequency")
plt.title("Histogram of Residuals for Scale 2-Model 0")
plt.show()
"""
resids60_0=[]
for i in range(0,len(group0preds)):
    #print("Prediction: "+str(group0preds[i][2])+" Actual: "+str(group0fitans[i][2]))
    resids60_0.append(group0preds[i][2]-group0fitans[i][2])
    mse0=mse0+(group0preds[i][2]-group0fitans[i][2])**2
"""
plt.hist(resids60_0)
plt.xlabel("Residuals")
plt.ylabel("Frequency")
plt.title("Histogram of Residuals for Scale 4-Model 0")
plt.show()
"""
#Residual plots group 1
resids57_1=[]
for i in range(0,len(group1preds)):
    #print("Prediction: "+str(group1preds[i][0])+" Actual: "+str(group1fitans[i][0]))
    resids57_1.append(group1preds[i][0]-group1fitans[i][0])
    mse1=mse1+(group1preds[i][0]-group1fitans[i][0])**2
"""
plt.hist(resids57_1)
plt.xlabel("Residuals")
plt.ylabel("Frequency")
plt.title("Histogram of Residuals for Scale 1-Model 1")
plt.show()
"""
resids58_1=[]
for i in range(0,len(group1preds)):
    print("Prediction: "+str(group1preds[i][1])+" Actual: "+str(group1fitans[i][1]))
    resids58_1.append(group1preds[i][1]-group1fitans[i][1])
    mse1=mse1+(group1preds[i][1]-group1fitans[i][1])**2
"""
plt.hist(resids58_1)
plt.xlabel("Residuals")
plt.ylabel("Frequency")
plt.title("Histogram of Residuals for Scale 2-Model 1")
plt.show()
"""
resids60_1=[]
for i in range(0,len(group1preds)):
    #print("Prediction: "+str(group1preds[i][2])+" Actual: "+str(group1fitans[i][2]))
    resids60_1.append(group1preds[i][2]-group1fitans[i][2])
    mse1=mse1+(group1preds[i][2]-group1fitans[i][2])**2
"""
plt.hist(resids60_1)
plt.xlabel("Residuals")
plt.ylabel("Frequency")
plt.title("Histogram of Residuals for Scale 4-Model 1")
plt.show()
"""
#Residual plots group 2
resids57_2=[]
for i in range(0,len(group2preds)):
    resids57_2.append(group2preds[i][0]-group2fitans[i][0])
    #print("Prediction: "+str(group0preds[i][0])+" Actual: "+str(group0fitans[i][0]))
    mse2=mse2+(group2preds[i][0]-group2fitans[i][0])**2
"""
plt.hist(resids57_2)
plt.xlabel("Residuals")
plt.ylabel("Frequency")
plt.title("Histogram of Residuals for Scale 1-Model 2")
plt.show()
"""
resids58_2=[]
for i in range(0,len(group2preds)):
    resids58_2.append(group2preds[i][1]-group2fitans[i][1])
    mse2=mse2+(group2preds[i][1]-group2fitans[i][1])**2
"""
plt.hist(resids58_2)
plt.xlabel("Residuals")
plt.ylabel("Frequency")
plt.title("Histogram of Residuals for Scale 2-Model 2")
plt.show()
"""
resids60_2=[]
for i in range(0,len(group2preds)):
    resids60_2.append(group2preds[i][2]-group2fitans[i][2])
    mse2=mse2+(group2preds[i][2]-group2fitans[i][2])**2
"""
plt.hist(resids60_2)
plt.xlabel("Residuals")
plt.ylabel("Frequency")
plt.title("Histogram of Residuals for Scale 4-Model 2")
plt.show()
"""
#Residual plots group 3
resids57_3=[]
for i in range(0,len(group3preds)):
    resids57_3.append(group3preds[i][0]-group3fitans[i][0])
    mse3=mse3+(group3preds[i][0]-group3fitans[i][0])**2
"""
plt.hist(resids57_3)
plt.xlabel("Residuals")
plt.ylabel("Frequency")
plt.title("Histogram of Residuals for Scale 1-Model 3")
plt.show()
"""
resids58_3=[]
for i in range(0,len(group3preds)):
    resids58_3.append(group3preds[i][1]-group3fitans[i][1])
    mse3=mse3+(group3preds[i][1]-group3fitans[i][1])**2
"""
plt.hist(resids58_3)
plt.xlabel("Residuals")
plt.ylabel("Frequency")
plt.title("Histogram of Residuals for Scale 2-Model 3")
plt.show()
"""
resids60_3=[]
for i in range(0,len(group3preds)):
    resids60_3.append(group3preds[i][2]-group3fitans[i][2])
    mse3=mse3+(group3preds[i][2]-group3fitans[i][2])**2


print((mse1+mse0+mse2+mse3)/(len(group0)+len(group1)+len(group2)+len(group3)))
install("XlsxWriter")
if len(group0tester)>0 and len(group0testerans)>0:
    writer = pd.ExcelWriter(str(nclus)+'example'+acti+'.xlsx', engine='xlsxwriter')
    coefs0=pd.DataFrame(data=group0model.coefs_[0])
    coefs0.to_excel(writer, 'Sheet1')
    writer.save()
if len(group1tester)>0 and len(group1testerans)>0:
    writer = pd.ExcelWriter(str(nclus)+'example2'+acti+'.xlsx', engine='xlsxwriter')
    coefs1=pd.DataFrame(data=group1model.coefs_[0])
    coefs1.to_excel(writer, 'Sheet2')
    writer.save()
if len(group2tester)>0 and len(group2testerans)>0:
    writer = pd.ExcelWriter(str(nclus)+'example3'+acti+'.xlsx', engine='xlsxwriter')
    coefs1=pd.DataFrame(data=group2model.coefs_[0])
    coefs1.to_excel(writer, 'Sheet3')
    writer.save()
if len(group3tester)>0 and len(group3testerans)>0:  
    writer = pd.ExcelWriter(str(nclus)+'example4'+acti+'.xlsx', engine='xlsxwriter')
    coefs1=pd.DataFrame(data=group3model.coefs_[0])
    coefs1.to_excel(writer, 'Sheet4')
    writer.save()

writer = pd.ExcelWriter('ofinterestnames.xlsx', engine='xlsxwriter')
of_interest2_names=[]
for i in of_interest2:
    of_interest2_names.append(column_names[i])
names=pd.DataFrame(data=of_interest2_names)
names.to_excel(writer, 'Sheet1')
writer.save()

#Analyze 3D relationships
#import thesis3d

#Standardize


    
