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

#Smooth periodograms and crossperiodograms
m=10

Z=linkage(X,'ward')
