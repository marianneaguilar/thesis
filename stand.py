#STANDARDIZE
data_numeric = []
    
for column in of_interest:
    temp=make_num(column)
    temp2 = pd.DataFrame({"A" : data[:,9],"B": temp})
    temp3 = pd.pivot_table(temp2, values='B',index=['A'],aggfunc=numpy.mean)
    temp4 = pd.pivot_table(temp2, values='B',index=['A'],aggfunc=numpy.std)
    for i in range(0,7200):
        person = data[i,9]
        sd = temp4.query('A == @person')
        mn = temp3.query('A == @person')
        if sd.B[0] > 0:
            temp[i]=(temp[i]-mn.B[0])/sd.B[0]
        else:
            temp[i]=numpy.nan
    data_numeric.append(temp)

data_numeric2 = [list(i) for i in zip(*data_numeric)]
data_numeric3 = pd.DataFrame(data_numeric2,columns=of_interest)
corrs=data_numeric3.corr()

list_names=[]
for i in of_interest:
    list_names.append(column_names[i])
ax2 = sns.heatmap(corrs,xticklabels=list_names,yticklabels=list_names)
plt.show()
