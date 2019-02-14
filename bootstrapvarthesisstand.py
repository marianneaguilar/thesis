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

import plotly.plotly as py
import plotly.graph_objs as go
install("seaborn")
import seaborn as sns

#Removed 17,55,57,58,59,60,61,62,63,64,65,66,125,126,127,128,129,130,131,132,133,134
of_interest = [4,5,6,7,14,15,18,19,20,21,22,23,24,25,79,111,112,113,114,115,117,118,136]


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
            data1.append(data_temp2[0:27])
    for row in data1:
        avg=numpy.mean(row)
        s=numpy.std(row)
        data_temp3=[]
        for i in row:
            data_temp3.append((i-avg)/s)
        data2.append(data_temp3)
    data3=[]
    person=0
    while person<238*len(of_interest):
        temp=[]
        for i in range(0,len(of_interest)):
            temp2=numpy.array(data2[person+i])
            if len(temp2) != 27:
                temp3=numpy.pad(temp2,(0,27-len(temp2)),'constant',constant_values=(0,0))
                temp.append(temp3)
            else:
                temp.append(temp2)
        temp=numpy.array(temp)
        data3.append(temp.flatten())
        person=person+len(of_interest)
    return data1,data2,data3

boot=0
install("sklearn")
from sklearn.cluster import FeatureAgglomeration
install('random')
import random

of_interest2 = [4,5,6,7,14,15,17,18,19,20,21,22,23,24,25,55,57,58,59,60,61,62,63,64,65,66,79,111,112,113,114,115,117,118,125,126,127,128,129,130,131,132,133,134,136]

while boot<99:
    rand_students=random.sample(range(238),150)
    prelimdata1c,prelimdata2c,prelimdata3c=fix(of_interest)
    data3c=[]
    for r in rand_students:
        data3c.append(prelimdata3c[r])
    acti='identity'
    nclus=4
    agglo=FeatureAgglomeration(n_clusters=nclus,affinity="euclidean")
    X=numpy.array(data3c)
    X=numpy.nan_to_num(X,copy=True)
    Z=agglo.fit(numpy.transpose(X))
    Z.labels_
    prelimdata1,prelimdata2,prelimdata3=fix(of_interest2)
    data1=[]
    data2=[]
    data3=[]
    for r in rand_students:
        data1.append(prelimdata1[r])
        data2.append(prelimdata2[r])
        data3.append(prelimdata3[r])
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
    from sklearn.neural_network import MLPRegressor
    neural0=MLPRegressor(hidden_layer_sizes=(10,10),activation=acti,solver='lbfgs',random_state=0)
    neural1=MLPRegressor(hidden_layer_sizes=(10,10),activation=acti,solver='lbfgs',random_state=0)
    neural2=MLPRegressor(hidden_layer_sizes=(10,10),activation=acti,solver='lbfgs',random_state=0)
    neural3=MLPRegressor(hidden_layer_sizes=(10,10),activation=acti,solver='lbfgs',random_state=0)
    lim0=int(numpy.trunc(len(group0)/2))
    lim1=int(numpy.trunc(len(group1)/2))
    lim2=int(numpy.trunc(len(group2)/2))
    lim3=int(numpy.trunc(len(group3)/2))
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
    if len(group1tester)>0 and len(group1testerans)>0:    
        X2=numpy.asarray(group1tester, dtype=float)
        X2=numpy.nan_to_num(X2,copy=True)
        X3=numpy.asarray(group1testerans, dtype=float)
        X3=numpy.nan_to_num(X3,copy=True)
        group1model=neural1.fit(X2,X3)
    if len(group2tester)>0 and len(group2testerans)>0:  
        X4=numpy.asarray(group2tester, dtype=float)
        X4=numpy.nan_to_num(X4,copy=True)
        X5=numpy.asarray(group2testerans,dtype=float)
        X5=numpy.nan_to_num(X5,copy=True)
        group2model=neural2.fit(X4,X5)
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
    if len(group0fit)>0:
        X8=numpy.asarray(group0fit,dtype=float)
        X8=numpy.nan_to_num(X8,copy=True)
        group0preds=group0model.predict(X8)
    if len(group1fit)>0:
        X9=numpy.asarray(group1fit,dtype=float)
        X9=numpy.nan_to_num(X9,copy=True)
        group1preds=group1model.predict(X9)
    if len(group2fit)>0:
        X10=numpy.asarray(group2fit,dtype=float)
        X10=numpy.nan_to_num(X10,copy=True)
        group2preds=group2model.predict(X10)
    if len(group3fit)>0:
        X11=numpy.asarray(group3fit,dtype=float)
        X11=numpy.nan_to_num(X11,copy=True)
        group3preds=group3model.predict(X11)
    mse0=0
    mse1=0
    mse2=0
    mse3=0
    for i in range(0,len(group0preds)):
        mse0=mse0+(group0preds[i][0]-group0fitans[i][0])**2
    for i in range(0,len(group0preds)):
        mse0=mse0+(group0preds[i][1]-group0fitans[i][1])**2
    for i in range(0,len(group0preds)):
        mse0=mse0+(group0preds[i][2]-group0fitans[i][2])**2
    for i in range(0,len(group1preds)):
        mse1=mse1+(group1preds[i][0]-group1fitans[i][0])**2
    for i in range(0,len(group1preds)):
        mse1=mse1+(group1preds[i][1]-group1fitans[i][1])**2
    for i in range(0,len(group1preds)):
        mse1=mse1+(group1preds[i][2]-group1fitans[i][2])**2
    for i in range(0,len(group2preds)):
        mse2=mse2+(group2preds[i][0]-group2fitans[i][0])**2
    for i in range(0,len(group2preds)):
        mse2=mse2+(group2preds[i][1]-group2fitans[i][1])**2
    for i in range(0,len(group2preds)):
        mse2=mse2+(group2preds[i][2]-group2fitans[i][2])**2
    for i in range(0,len(group3preds)):
        mse3=mse3+(group3preds[i][0]-group3fitans[i][0])**2
    for i in range(0,len(group3preds)):
        mse3=mse3+(group3preds[i][1]-group3fitans[i][1])**2
    for i in range(0,len(group3preds)):
        mse3=mse3+(group3preds[i][2]-group3fitans[i][2])**2
    print((mse1+mse0+mse2+mse3)/((len(group0)+len(group1)+len(group2)+len(group3))))
    install("XlsxWriter")
    if len(group0tester)>0 and len(group0testerans)>0:
        writer = pd.ExcelWriter(str(nclus)+'example'+acti+'iteration'+str(boot)+'.xlsx', engine='xlsxwriter')
        coefs0=pd.DataFrame(data=group0model.coefs_[0])
        coefs0.to_excel(writer, 'Sheet1')
        writer.save()
    if len(group1tester)>0 and len(group1testerans)>0:
        writer = pd.ExcelWriter(str(nclus)+'example2'+acti+'iteration'+str(boot)+'.xlsx', engine='xlsxwriter')
        coefs1=pd.DataFrame(data=group1model.coefs_[0])
        coefs1.to_excel(writer, 'Sheet2')
        writer.save()
    if len(group2tester)>0 and len(group2testerans)>0:
        writer = pd.ExcelWriter(str(nclus)+'example3'+acti+'iteration'+str(boot)+'.xlsx', engine='xlsxwriter')
        coefs1=pd.DataFrame(data=group2model.coefs_[0])
        coefs1.to_excel(writer, 'Sheet3')
        writer.save()
    if len(group3tester)>0 and len(group3testerans)>0:  
        writer = pd.ExcelWriter(str(nclus)+'example4'+acti+'iteration'+str(boot)+'.xlsx', engine='xlsxwriter')
        coefs1=pd.DataFrame(data=group3model.coefs_[0])
        coefs1.to_excel(writer, 'Sheet4')
        writer.save()
    boot=boot+1

writer = pd.ExcelWriter('ofinterestnames.xlsx', engine='xlsxwriter')
of_interest2_names=[]
for i in of_interest2:
    of_interest2_names.append(column_names[i])
names=pd.DataFrame(data=of_interest2_names)
names.to_excel(writer, 'Sheet1')
writer.save()
