#Analyze 3D relationships
from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure()
ax3 = Axes3D(fig)
ax3.plot(xs=x,ys=make_num(56),zs=make_num(20))
ax3.set_xlabel('Day of month')
ax3.set_ylabel('Mood')
ax3.set_zlabel('Sleep time')
plt.show()

#Analyze individual students in 3D plot

install("colour")
from colour import Color
def analyze_relationships_3d(col1, col2):
    red = Color("red")
    colors = list(red.range_to(Color("violet"),len(indivs)+1))
    indivs2 = list(set(data[:,9]))
    for pos in range(0,(len(indivs2)-1)):
        indiv=indivs2[pos]
        temp_xs=data[data[:,9]==indiv,2]
        temp_mood=data[data[:,9]==indiv,col1]
        temp_sleep=data[data[:,9]==indiv,col2]
        xse=[]
        for entry in temp_xs:
            try:
                xse.append(float(entry))
            except:
                xse.append(numpy.nan)
        mood=[]
        for entry in temp_mood:
            try:
                mood.append(float(entry))
            except:
                mood.append(numpy.nan)
        sleep=[]
        for entry in temp_sleep:
            try:
                sleep.append(float(entry))
            except:
                sleep.append(numpy.nan)
        ax4.plot(xs=xse,ys=mood,zs=sleep,c=colors[pos+1].hex)

fig=plt.figure()
ax4 = Axes3D(fig)
analyze_relationships_3d(56,20)        
ax4.set_xlabel('Day of month')
ax4.set_ylabel('Mood')
ax4.set_zlabel('Sleep time')
plt.show()
