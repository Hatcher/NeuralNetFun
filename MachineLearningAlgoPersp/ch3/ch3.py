import os
import matplotlib.pyplot as pl
import numpy as np
import pcn

os.chdir(os.path.join(os.path.dirname(__file__), 'pimaData/'))
pima = np.loadtxt('pima-indians-diabetes.data', delimiter=',')
#np.shape(pima)
indices0 = np.where(pima[:,8] == 0) #Where the class is 0
indices1 = np.where(pima[:,8] == 1) #Where the class is 1


pl.plot(pima[indices0,0],pima[indices0,1],'go')
pl.plot(pima[indices1,0],pima[indices1,1],'rx')

p = pcn.pcn(pima[:,:8], pima[:,8:9])
p.pcntrain(pima[:,:8],pima[:,8:9],0.25,100)
p.confmat(pima[:,:8],pima[:,8:9])

trainin = pima[::2,:8]
testin = pima[1::2,:8]
traintgt = pima[::2, 8:9]
testtgt = pima[1::2,8:9]



pl.show()
