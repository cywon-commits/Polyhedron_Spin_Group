import sys,numpy as np,time
sys.path.insert(0,'/tmp/s4'); from core import solve, classify
k1s=np.round(np.arange(-1.0,2.51,0.07),3); k2s=np.round(np.arange(-1.5,1.51,0.07),3)
lab=np.empty((len(k2s),len(k1s)),dtype='U2')
t=time.time()
for i,k2 in enumerate(k2s):
    for j,k1 in enumerate(k1s):
        lab[i,j]=classify(solve([k1,k2,-1.0,0.0])[1])
np.savez('grid_kappa.npz',k1=k1s,k2=k2s,lab=lab); print(time.time()-t, np.unique(lab,return_counts=True))
