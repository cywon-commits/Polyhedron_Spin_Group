"""Fusion rules of 2D point defects (pi_1 conjugacy classes) for the non-Abelian phases."""
from style import *
import json, os, numpy as np
from fractions import Fraction as F
from matplotlib.patches import Rectangle
D=json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)),'..','data','fusion.json')))
def tex(l):
    m={'-1':r'$-1$','4a':r'$4_a$','4b':r'$4_b$'}
    if l in m: return m[l]
    if l[-1] in '+-': return r'$%s^{%s}$'%(l[:-1],'+' if l[-1]=='+' else '-')
    return r'$%s$'%l
def frac(s):
    f=F(s); return '1' if f==1 else '%d/%d'%(f.numerator,f.denominator)
ATT,REP,NEU='#1a7f37','#c0392b','0.25'
BG={1:'white',2:'#dde8f5',3:'#b7cde9'}
def table(ax,g,title):
    d=D[g]; L=d['labels'][1:]; th=dict(zip(d['labels'],d['theta_deg'])); n=len(L)
    for i,a in enumerate(L):
        for j,b in enumerate(L):
            P=d['P'][i+1][j+1]; y=n-1-i
            ax.add_patch(Rectangle((j,y),1,1,fc=BG[len(P)],ec='0.55',lw=0.6))
            if '1' in P: ax.add_patch(Rectangle((j+0.04,y+0.04),0.92,0.92,fc='none',ec='k',lw=1.3))
            items=list(P.items()); m=len(items)
            for k,(c,p) in enumerate(items):
                e=(th[c]/180)**2-(th[a]/180)**2-(th[b]/180)**2
                col=ATT if e<-1e-9 else (REP if e>1e-9 else NEU)
                yy=y+0.5+(m-1-2*k)*(0.16 if m==3 else 0.17)
                ax.text(j+0.5,yy,tex(c)+(' '+frac(p) if m>1 else ''),ha='center',va='center',fontsize=8.2 if m==3 else 9.0,color=col)
    ax.set_xlim(0,n); ax.set_ylim(0,n)
    ax.set_xticks(np.arange(n)+0.5); ax.set_xticklabels([tex(l) for l in L],fontsize=9.5)
    ax.set_yticks(np.arange(n)+0.5); ax.set_yticklabels([tex(l) for l in L[::-1]],fontsize=9.5)
    ax.xaxis.tick_top(); ax.tick_params(length=0,pad=2)
    for s in ax.spines.values(): s.set_visible(False)
    ax.set_title(title,loc='left',fontsize=9.5,pad=16)
fig=plt.figure(figsize=(6.9,6.0))
gs=fig.add_gridspec(2,2,width_ratios=[6.5,4.9],height_ratios=[3.9,1.3],hspace=0.52,wspace=0.17)
table(fig.add_subplot(gs[0,0]),'2T',r'(a) T phase: $\pi_1=2T$ (24 elements)')
table(fig.add_subplot(gs[0,1]),'Dic3',r'(b) P/222 phases: $\pi_1=\mathrm{Dic}_3$ (12)')
fig.text(0.5,0.335,'green: attractive channel;   red: repulsive;   framed: annihilation possible;   shading: number of channels',ha='center',fontsize=9)
# (c) annihilation probability
ax=fig.add_subplot(gs[1,:])
groups=[(r'C: $\mathbb{Z}_2$',[('1/2',r'$\pi$','1')]),
        (r'U, D: $\mathbb{Z}_4$',[('',r'$\pi$','1'),('',r'$2\pi$','1'),('',r'$3\pi$','1')]),
        (r'P/222: $\mathrm{Dic}_3$',None),(r'T: $2T$',None),(r'O: $2O$',None)]
src={r'P/222: $\mathrm{Dic}_3$':'Dic3',r'T: $2T$':'2T',r'O: $2O$':'2O'}
x=0; xt=[]; xl=[]
for name,items in groups:
    if items is None:
        d=D[src[name]]; items=[(l,None,p) for l,p in zip(d['labels'][1:],d['annihilation_prob'][1:])]
    x0=x
    for l,_,p in items:
        v=float(F(p)); ax.plot([x,x],[0,v],color='0.6',lw=1.0); ax.plot(x,v,'o',ms=5.5,color=C['T'] if v<1 else '0.35')
        x+=1
    xt.append((x0+x-1)/2); xl.append(name); ax.axvline(x+0.0,color='0.85',lw=0.8); x+=1
ax.set_xticks(xt); ax.set_xticklabels(xl,fontsize=9)
ax.set_xlim(-0.8,x-1.2); ax.set_ylim(0,1.12); ax.set_yticks([0,0.5,1])
ax.set_ylabel(r'$P(a\,\bar a\to1)$')
ax.set_title(r'(c) annihilation probability of a defect with its antidefect (one dot per class)',loc='left',fontsize=9.5)
fig.savefig(os.path.join(os.path.dirname(os.path.abspath(__file__)),'..','fig','fig_fusion.pdf'))
