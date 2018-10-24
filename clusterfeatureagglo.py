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
        temp2=numpy.array(data2_per[person+i])
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
agglo=FeatureAgglomeration(n_clusters=6,affinity="euclidean")
X=numpy.array(data3)
X=numpy.nan_to_num(X,copy=True)
Z=agglo.fit(X)
Z.labels_
