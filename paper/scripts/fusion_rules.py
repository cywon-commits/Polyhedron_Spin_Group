import numpy as np, itertools
def qmul(a,b):
    a0,a1,a2,a3=a; b0,b1,b2,b3=b
    return np.array([a0*b0-a1*b1-a2*b2-a3*b3, a0*b1+a1*b0+a2*b3-a3*b2, a0*b2-a1*b3+a2*b0+a3*b1, a0*b3+a1*b2-a2*b1+a3*b0])
def qinv(a): return a*np.array([1,-1,-1,-1])
def close(gens):
    G=[np.array([1.,0,0,0])]
    changed=True
    while changed:
        changed=False
        for g in list(G):
            for h in gens:
                p=qmul(g,h)
                if not any(np.allclose(p,x,atol=1e-9) for x in G): G.append(p); changed=True
    return G
def binary(name):
    if name=='2T': gens=[np.array([0,1,0,0.]),np.array([.5,.5,.5,.5])]
    if name=='2O': gens=[np.array([1,1,0,0])/np.sqrt(2),np.array([.5,.5,.5,.5])]
    if name=='Dic3': gens=[np.array([np.cos(np.pi/3),0,0,np.sin(np.pi/3)]),np.array([0,1,0,0.])]
    return close(gens)
def idx(G,p): return [i for i,x in enumerate(G) if np.allclose(x,p,atol=1e-9)][0]
def classes(G):
    n=len(G); seen=set(); C=[]
    for i in range(n):
        if i in seen: continue
        c=sorted({idx(G,qmul(qmul(g,G[i]),qinv(g))) for g in G}); seen|=set(c); C.append(c)
    return C
def info(G,c):
    q=G[c[0]]; th=2*np.arccos(np.clip(q[0],-1,1))
    # order
    p=q.copy(); k=1
    while not np.allclose(p,[1,0,0,0],atol=1e-9): p=qmul(p,q); k+=1
    return th,k
def class_algebra(G,C):
    cid={i:a for a,c in enumerate(C) for i in c}
    N=np.zeros((len(C),len(C),len(C)),int)
    for a,ca in enumerate(C):
        for b,cb in enumerate(C):
            x=G[ca[0]]  # fix representative of a, sum over class b -> multiplicity / normalization
            cnt={}
            for j in cb:
                r=cid[idx(G,qmul(x,G[j]))]; cnt[r]=cnt.get(r,0)+1
            for r,v in cnt.items(): N[a,b,r]=v*len(ca)//len(C[r])
    return N
from fractions import Fraction as F
import json, os
LABELS={'2T':['1','6+','6-','4','3+','3-','-1'],'Dic3':['1','6','4a','4b','3','-1'],'2O':['1','8+','6','4a','4b','3','8-','-1']}
def fusion(name):
    """Fusion of 2D point defects labelled by conjugacy classes of pi_1 = name.
    P[a][b][c]: probability that defects of classes a and b fuse into class c when the
    relative holonomy is uniform (representative of a fixed, b uniform over its class)."""
    G=binary(name); C=classes(G); I=[info(G,c) for c in C]
    o=np.argsort([t+0.001*len(c) for (t,k),c in zip(I,C)]); C=[C[i] for i in o]; I=[I[i] for i in o]
    lab=LABELS[name]; cid={i:a for a,c in enumerate(C) for i in c}
    P=[[None]*len(C) for _ in C]
    for a,ca in enumerate(C):
        for b,cb in enumerate(C):
            cnt={}
            for j in cb:
                r=cid[idx(G,qmul(G[ca[0]],G[j]))]; cnt[r]=cnt.get(r,0)+1
            P[a][b]={lab[r]:str(F(v,len(cb))) for r,v in sorted(cnt.items())}
    return dict(labels=lab,size=[len(c) for c in C],order=[int(k) for t,k in I],
                theta_deg=[round(float(np.degrees(t))) for t,k in I],
                inverse=[lab[cid[idx(G,qinv(G[c[0]]))]] for c in C],
                annihilation_prob=[str(F(1,len(c))) for c in C],P=P)
if __name__=='__main__':
    out={n:fusion(n) for n in LABELS}
    path=os.path.join(os.path.dirname(os.path.abspath(__file__)),'..','data','fusion.json')
    json.dump(out,open(path,'w'),indent=1); print('wrote',path)
