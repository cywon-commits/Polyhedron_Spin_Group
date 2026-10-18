from style import *; import numpy as np
from matplotlib.colors import ListedColormap
from matplotlib.patches import Circle
order=['T','U','D','C','F','?']; cmap=ListedColormap([C['T'],C['U'],C['D'],C['C'],C['F'],'#DDDDDD'])
def idx(lab):
    m={'T':0,'U':1,'u':1,'D':2,'C':3,'F':4}
    return np.vectorize(lambda s:m.get(s,5))(lab)
def lookup(xs,ys,L,x,y):
    i=np.argmin(abs(ys-y)); j=np.argmin(abs(xs-x)); return L[i,j]
def seg(ax,xs,ys,L,line,pair,param,lo,hi,eps,**kw):
    t=np.linspace(lo,hi,600); X,Y=line(t)
    # normal offset
    dX=np.gradient(X); dY=np.gradient(Y); nrm=np.hypot(dX,dY); nx,ny=-dY/nrm,dX/nrm
    ok=[]
    for x,y,a,b in zip(X,Y,nx,ny):
        s={lookup(xs,ys,L,x+eps*a,y+eps*b),lookup(xs,ys,L,x-eps*a,y-eps*b)}
        ok.append(s==set(pair))
    ok=np.array(ok)
    ax.plot(np.where(ok,X,np.nan),np.where(ok,Y,np.nan),**kw)
fig=plt.figure(figsize=(DCOL,4.9))
gs=fig.add_gridspec(2,2,height_ratios=[3.0,1.25],hspace=0.55,wspace=0.28)
# (a) kappa plane
d=np.load('grid_kappa.npz'); k1,k2,L=d['k1'],d['k2'],d['lab']; I=idx(L)
Lm=np.vectorize(lambda s:{'u':'U'}.get(s,s))(L)
ax=fig.add_subplot(gs[0,0])
ax.imshow(I,origin='lower',extent=[k1[0]-0.035,k1[-1]+0.035,k2[0]-0.035,k2[-1]+0.035],cmap=cmap,vmin=-0.5,vmax=5.5,aspect='auto',interpolation='nearest',alpha=0.85)
kw=dict(color='k',lw=1.6)
seg(ax,k1,k2,Lm,lambda t:(4/9-t,t),('T','F'),None,-1.5,1.5,0.06,**kw)
seg(ax,k1,k2,Lm,lambda t:(0.375-0.75*t-0.225*t**2,t),('D','F'),None,-1.5,1.5,0.06,ls='--',**{k:v for k,v in kw.items()})
seg(ax,k1,k2,Lm,lambda t:(6+3*t,t),('C','D'),None,-1.5,-0.9,0.06,ls=':',**kw)
for s,(x,y) in {'T':(1.6,0.9),'U':(1.15,-0.05),'D':(1.5,-0.95),'F':(-0.55,-0.6),'C':(2.2,-1.38),'U$^\\prime$':(-0.85,1.32)}.items():
    ax.text(x,y,s,fontsize=11,ha='center',va='center',color='white',fontweight='bold')
ax.set_xlabel(r'$\kappa_1/|\kappa_3|$'); ax.set_ylabel(r'$\kappa_2/|\kappa_3|$')
ax.set_title(r'(a) channel space ($n=4$, $\kappa_4=0$)',loc='left',fontsize=9.5)
# (b) physical plane
d=np.load('grid_phys.npz'); G,H,L=d['G'],d['H'],d['lab']; I=idx(L)
Lm=np.vectorize(lambda s:{'u':'U'}.get(s,s))(L)
ax=fig.add_subplot(gs[0,1])
ax.imshow(I,origin='lower',extent=[G[0]-0.025,G[-1]+0.025,H[0]-0.025,H[-1]+0.025],cmap=cmap,vmin=-0.5,vmax=5.5,aspect='auto',interpolation='nearest',alpha=0.85)
seg(ax,G,H,Lm,lambda t:(t,0.45+0.3*t),('T','C'),None,0,2.2,0.04,**kw)
seg(ax,G,H,Lm,lambda t:((4/3-20/27*t)*9/7,t),('T','F'),None,0,1.2,0.04,ls='--',**kw)
seg(ax,G,H,Lm,lambda t:(1+0*t,t),('C','F'),None,0,1.2,0.04,ls=':',**kw)
for s,(x,y) in {'T':(0.7,0.22),'C':(0.45,0.95),'F':(1.8,0.6),'U':(1.1,0.69)}.items():
    ax.text(x,y,s,fontsize=11,ha='center',va='center',color='white',fontweight='bold')
ax.set_xlabel(r'$zG$'); ax.set_ylabel(r'$zH_4$')
ax.set_title(r'(b) physical plane ($J=1$, $K=0.5$)',loc='left',fontsize=9.5)
# (c) configuration strip
ax=fig.add_subplot(gs[1,:]); ax.axis('off'); ax.set_aspect('equal')
def arrows(cx,vecs,mult,name,col):
    ax.add_patch(Circle((cx,0),0.95,fill=False,ls='--',lw=0.6,color='0.6'))
    for v,m in zip(vecs,mult):
        v=np.array(v)/np.linalg.norm(v)*0.85
        if m==1:
            ax.annotate('',xy=(cx+v[0],v[1]),xytext=(cx,0),arrowprops=dict(arrowstyle='-|>',lw=1.6,color=col,mutation_scale=10))
        else:
            p=np.array([-v[1],v[0]])*0.07
            for s in (1,-1):
                ax.annotate('',xy=(cx+v[0]+s*p[0],v[1]+s*p[1]),xytext=(cx+s*p[0],s*p[1]),arrowprops=dict(arrowstyle='-|>',lw=1.3,color=col,mutation_scale=9))
            ax.text(cx+1.18*v[0],1.18*v[1],r'$\times%d$'%m,fontsize=8,ha='center',va='center')
    ax.text(cx,-1.25,name,ha='center',va='top',fontsize=9)
a=np.radians
P=np.array([[0.3,0.95],[0.9,-0.45],[-0.95,-0.35],[-0.25,0.15]])
arrows(0,P,[1,1,1,1],'T\n$M=0$','#4C72B0')
arrows(4,[[0,1],[np.cos(a(-30)),np.sin(a(-30))],[np.cos(a(210)),np.sin(a(210))]],[2,1,1],r'U $\{211\}$'+'\n'+r'$M=n/4$','#DD8452')
arrows(8,[[np.cos(a(70)),np.sin(a(70))],[np.cos(a(-75)),np.sin(a(-75))]],[2,2],r'D bent $\{22\}$'+'\n'+r'$0<M<n$','#55A868')
arrows(12,[[0,1],[0,-1]],[2,2],r'C $\{22\}$'+'\n'+r'$M=0$','#C44E52')
arrows(16,[[0,1]],[4],'F\n$M=n$','#8172B3')
arrows(20,[[0,1],[0,-1]],[3,1],r'M2 $\{31\}$'+'\n'+r'$M=n/2$','#E5AE38')
ax.set_xlim(-1.6,21.6); ax.set_ylim(-2.3,1.3)
ax.set_title('(c) cluster states (2D sketches; '+r'$\times k$'+' = $k$ coincident spins)',loc='left',fontsize=9.5)
fig.savefig('fig/fig_T0.pdf')
