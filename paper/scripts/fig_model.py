from style import *; import numpy as np
from matplotlib.patches import Circle, FancyArrowPatch
fig=plt.figure(figsize=(DCOL,2.55))
gs=fig.add_gridspec(1,3,width_ratios=[1,1.35,1.15],wspace=0.42)
# (a) one cluster: tetrahedron projected
ax=fig.add_subplot(gs[0]); ax.set_aspect('equal'); ax.axis('off')
tet=np.array([[1,1,1],[1,-1,-1],[-1,1,-1],[-1,-1,1]])/np.sqrt(3)
Rz=lambda a:np.array([[np.cos(a),-np.sin(a),0],[np.sin(a),np.cos(a),0],[0,0,1]])
Rx=lambda a:np.array([[1,0,0],[0,np.cos(a),-np.sin(a)],[0,np.sin(a),np.cos(a)]])
P=(Rx(-0.55)@Rz(0.35)@tet.T).T
cols=['#4C72B0','#C44E52','#55A868','#8172B3']
ax.add_patch(Circle((0,0),1.05,fill=False,ls='--',lw=0.7,color='0.6'))
for i,p in enumerate(P):
    ax.annotate('',xy=(p[0],p[1]),xytext=(0,0),arrowprops=dict(arrowstyle='-|>',lw=1.8,color=cols[i],mutation_scale=12))
    ax.text(1.22*p[0],1.22*p[1],r'$\mathbf{S}_{%d}$'%(i+1),ha='center',va='center',fontsize=10,color=cols[i])
for i in range(4):
    for j in range(i+1,4):
        ax.plot([P[i,0],P[j,0]],[P[i,1],P[j,1]],color='0.75',lw=0.6,zorder=0)
ax.set_xlim(-1.5,1.5); ax.set_ylim(-1.55,1.45)
ax.text(0,-1.5,r'intra: $J\,x+K\,x^2$ (AF)',ha='center',fontsize=9)
ax.set_title('(a) spin body ($n=4$)',loc='left',fontsize=9.5)
# (b) lattice of clusters
ax=fig.add_subplot(gs[1]); ax.set_aspect('equal'); ax.axis('off')
rng=np.random.default_rng(2)
for X in range(3):
    for Y in range(2):
        cx,cy=2.2*X,2.2*Y
        ax.add_patch(Circle((cx,cy),0.78,color='#EEF2F8',ec='0.6',lw=0.6))
        R=Rz(0.25*X)@Rx(-1.1)@Rz(0.5)
        Q=(R@tet.T).T*0.62
        for i,q in enumerate(Q):
            ax.annotate('',xy=(cx+q[0],cy+q[1]),xytext=(cx,cy),arrowprops=dict(arrowstyle='-|>',lw=1.1,color=cols[i],mutation_scale=7))
        if X<2: ax.plot([cx+0.8,cx+1.4],[cy,cy],color='k',lw=1.6)
        if Y<1: ax.plot([cx,cx],[cy+0.8,cy+1.4],color='k',lw=1.6)
ax.set_xlim(-1,5.4); ax.set_ylim(-1.3,3.1)
ax.text(2.2,-1.25,r'inter: $-\sum_{i,j}[G x^3+H_4x^4]$ (FM)',ha='center',fontsize=9)
ax.set_title('(b) lattice of spin bodies',loc='left',fontsize=9.5)
# (c) channel spectrum
ax=fig.add_subplot(gs[2])
J,K,zG,zH=1,0.5,1.0,0.5
f=[J,2*K/3,0,0]; g=[0.6*zG,4*zH/7,0.4*zG,8*zH/35]
l=np.arange(1,5); kap=np.array(f)-np.array(g)
ax.bar(l-0.2,f,0.38,color='0.75',label=r'$f_l$ (intra)')
ax.bar(l+0.2,-np.array(g),0.38,color=['#8172B3','#C44E52','#8172B3','#C44E52'],label=r'$-zg_l$ (inter)')
ax.plot(l,kap,'ko',ms=5,label=r'$\kappa_l=f_l-zg_l$')
ax.axhline(0,color='k',lw=0.6)
ax.set_xticks(l); ax.set_xlabel(r'channel $l$'); ax.set_ylabel('coefficient')
ax.set_ylim(-0.75,1.65)
ax.set_xticklabels(['1\nodd','2\neven','3\nodd','4\neven'])
ax.legend(loc='upper right',fontsize=8.3,handlelength=1.0)
ax.set_title(r'(c) channel spectrum',loc='left',fontsize=9.5)
fig.savefig('fig/fig_model.pdf')
