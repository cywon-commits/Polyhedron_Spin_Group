import sys,numpy as np,time
sys.path.insert(0,'/tmp/s4'); from core import solve, classify
J,K=1.0,0.5
Gs=np.round(np.arange(0.05,2.21,0.05),3); Hs=np.round(np.arange(0.0,1.21,0.05),3)
lab=np.empty((len(Hs),len(Gs)),dtype='U2')
t=time.time()
for i,H in enumerate(Hs):
    for j,G in enumerate(Gs):
        lab[i,j]=classify(solve([J-0.6*G,2*K/3-4*H/7,-0.4*G,-8*H/35])[1])
np.savez('grid_phys.npz',G=Gs,H=Hs,lab=lab); print(time.time()-t,np.unique(lab,return_counts=True))
