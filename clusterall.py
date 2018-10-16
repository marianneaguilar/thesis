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
comparisons=int(t3/len(individuals))
data2_per_sm_usable=numpy.transpose(data2_per_sm2)
for col in range(0,p3):
    temp_col=data2_per_sm_usable[col]
    person1=0
    while person1 < t3-comparisons:
        reorganize1=[]
        r1=0
        while r1 < comparisons:
            reorganize1.append(temp_col[person1+r1:person1+r1+len(of_interest)])
            r1=r1+len(of_interest)
        person2=person1+comparisons
        while person2 < t3:
            reorganize2=[]
            r2=0
            while r2 < comparisons:
                reorganize2.append(temp_col[person2+r2:person2+r2+len(of_interest)])
                r2=r2+len(of_interest)
            count=0
            divisor=0
            try:
                e1=scipy.linalg.eigvals(numpy.array(reorganize1))
            except:
                e1=[1,1]
            Q_new1=1
            for i in e1:
                Q_new1=Q_new1*i/(10**(-18))
            if math.isnan(Q_new1):
                Q_new1 = 0
            if Q_new1 != 0:
                divisor=math.trunc(math.log10(abs(Q_new1)))+divisor
                count=count+1
            try:
                e2=scipy.linalg.eigvals(numpy.array(reorganize2))
            except:
                e2=[1,1]
            Q_new2=1
            for i in e2:
                Q_new2=Q_new2*i/(10**(-18))
            if math.isnan(Q_new2):
                Q_new2 = 0
            if Q_new2 != 0:
                divisor=math.trunc(math.log10(abs(Q_new2)))+divisor
                count=count+1
            try:
                e3=scipy.linalg.eigvals(numpy.matrix(reorganize1)+numpy.matrix(reorganize2))
            except:
                e3=[1,1]
            Q_new12=1
            for i in e3:
                Q_new12=Q_new12*i/(10**(-18))
            if Q_new12 != 0:
                divisor=math.trunc(math.log10(abs(Q_new12)))+divisor
                count=count+1
            avgdiv=0
            if count != 0:
                avgdiv=divisor/count
            Q_new1=Q_new1/(10**avgdiv)
            Q_new2=Q_new2/(10**avgdiv)
            Q_new12=Q_new12/(10**avgdiv)
            if Q_new12 == 0 or math.isnan(Q_new12):
                Q_new=(2**(2*(2*m+1)))*(Q_new1**(2*m+1)*Q_new2**(2*m+1))
            else:
                Q_new=(2**(2*(2*m+1)))*((Q_new1**(2*m+1))*(Q_new2**(2*m+1)))/(Q_new12**(2*(2*m+1)))
            if math.isnan(Q_new):
                op=op+1
            Qxyw.append(Q_new)
            person2=person2+comparisons
        person1=person1+comparisons
Qxy=[]
num_pairs=sum(x for x in range(1,238))
pairs=0
while pairs < num_pairs:
    s=0
    for w in range(0,14):
        s=s+Qxyw[pairs+w*num_pairs]
    Qxy.append(s/15)
    pairs=pairs+1
Z=linkage(Qxy,'single',optimal_ordering=True)
plt.title('Hierarchical Clustering Dendrogram of Multivariate Time Series')
dendrogram(Z,labels=individuals)
#dendrogram(Z,truncate_mode='lastp',p=6,show_contracted=True)
