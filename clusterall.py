import scipy
install("statistics")
import statistics
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.signal import periodogram,csd
install("math")
import math
#N=number time series objects
#T=length of time serie objects (using 30 days here)
N=len(individuals)
T=30
P=N
#Reorganize data into time series format without using pivot function
data2=[]
for i in individuals:
    for col in of_interest:
        data_temp=data[i==data[:,9],col]
        data_temp2=[]
        for entry in range(0,len(data_temp)-1):
            try:
                data_temp2.append(float(data_temp[entry]))
            except:
                a=0
                c=0
                if entry-14>=0:
                    try:
                        a=a+data_temp[entry-14]
                        c=c+1
                    except:
                        a=a+0
                        c=c+0
                if entry-7>=0:
                    try:
                        a=a+data_temp[entry-7]
                        c=c+1
                    except:
                        a=a+0
                        c=c+0
                if entry+7<len(temp_data2):
                    try:
                        a=a+data_temp[entry+7]
                        c=c+1
                    except:
                        a=a+0
                        c=c+0
                if entry+14<len(temp_data2):
                    try:
                        a=a+data_temp[entry+14]
                        c=c+1
                    except:
                        a=a+0
                        c=c+0
                if c==0:
                    temp_data2.append(numpy.mean(temp_data2))
                else:
                    temp_data2.append(a/c)
        data2.append(data_temp2[0:29])
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
m=5
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

#Compute likelihood test for each frequency        
Qxyw=[]
comparisons=t3/len(individuals)
for person1 in range(0,p3-1):
    determinants=[]
    p=0
    while p < len(data2_per_sm)-1):
        s=0
        comparisons.append(individuals[q]+"+"+individuals[p])
        for freq in range(0,p2-1):
            Q_new1=abs(data2_per_sm[p,freq]**2)
            Q_new2=abs(data2_per_sm[q,freq]**2)
            Q_new12=abs((data2_per_sm[p,freq]+data2_per_sm[q,freq])**2)
            Q_new=(2**(2*2))*(Q_new1*Q_new2)/(Q_new12**2)
            s=s+Q_new
        Qxyw.append(s/p2)
Z=linkage(Qxyw,'single',optimal_ordering=True)
plt.title('Hierarchical Clustering Dendrogram of Multivariate Time Series')
dendrogram(Z,labels=individuals)
dendrogram(Z,truncate_mode='lastp',p=6,show_contracted=True)
