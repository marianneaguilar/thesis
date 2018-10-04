import spicy
install("statistics")
import statistics
from scipy.cluster.hierarchy import dendrogram, linkage
from spicy.signal import periodogram
#N=number time series objects
#T=length of time serie objects (using 30 days here)
N=length(list(set(data[:,9])))
T=30
P=N
#Reorganize data into time series format without using pivot function
data2=[]
for i in list(set(data[:,9])):
    for col in of_interest:
        data_temp=data[i==data[:,9],col]
        data2.append(data_temp[0:29])
#data2_fft = multivariate fourier transform of time series
data2_fft = numpy.fft.fftn(data2)
#Option 2:skip to periodogram
data2_per = []
for pos in range(0,len(data2[:,1])-1):
    data2_per.append(periodogram(data2[pos,:]))
#Calculate cross periodograms
data2_cper=[]
loc=0
while loc < len(data2[:,1]):
    for i in range(0,len(of_interest)-1):
        for j in range(0,len(of_interest)-1):
            data2_cper.append(spicy.signal.csd(data2[loc+i,:],data2[loc+j,:]))
    loc=loc+len(of_interest)

t2=len(data2_per[:,1])
p2=len(data2_per[1,:])
t3=len(data2_cper[:,1])
p3=len(data2_cper[1,:])

#Smooth periodograms and crossperiodograms
m=10
data2_per_sm=[]
data2_cper_sm=[]
for col in range(0,t2-1):
    if col < m:
        s = 0
        for i in range(0,col+m):
            s=s+sum(data[i,:])
        data2_per_sm.append(s)
    else if col > (t2-1-m):
        s2 = 0
        for i in range(col-m,t2-1):
            s2=s2+sum(data[i,:])
        data2_per_sm.append(s2)
    else:
        s3 = 0
        for i in range(col-m,col+m):
            s3=s3+sum(data[i,:])
        data2_per_sm.append(s3)

for col in range(0,t3-1):
    if col < m:
        s = 0
        for i in range(0,col+m):
            s=s+sum(data[i,:])
        data2_cper_sm.append(s)
    else if col > (t3-1-m):
        s2 = 0
        for i in range(col-m,t3-1):
            s2=s2+sum(data[i,:])
        data2_cper_sm.append(s2)
    else:
        s3 = 0
        for i in range(col-m,col+m):
            s3=s3+sum(data[i,:])
        data2_cper_sm.append(s3)

#Compute likelihood test for each frequency        
Qxyw=[]
comparisons=[]
for q in range(0,len(data2_per_sm[:,1])-2):
    for p in range(q+1,len(data2_per_sm[:,1])-1):
        comparisons.append(individuals[q]+"+"+individuals[p])
        
Z=linkage(X,'ward')
