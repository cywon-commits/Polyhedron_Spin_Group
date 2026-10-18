from style import *; import numpy as np
fig,axs=plt.subplots(1,2,figsize=(DCOL,2.35))
q=np.linspace(0,1.25,400)
# l=2 (cubic invariant): f = r q^2 - w q^3 + u q^4, w=1,u=1 ; r_c = 2/9 (=w^2/(4u)*... ) 
w,u=1.0,1.0; rc=w**2/(4*u)
for r,ls,lab in [(0.32,'-','$T>T_c$'),(rc,'--','$T=T_c$'),(0.15,':','$T<T_c$')]:
    axs[0].plot(q,r*q**2-w*q**3+u*q**4,ls,color='#C44E52',label=lab)
axs[0].set_title(r'(a) $l=2$ (nematic): cubic invariant $\Rightarrow$ 1st order',loc='left',fontsize=9.3)
axs[0].set_ylim(-0.02,0.07); axs[0].set_xlim(0,1.0)
for r,ls,lab in [(0.06,'-','$T>T_c$'),(0.0,'--','$T=T_c$'),(-0.06,':','$T<T_c$')]:
    axs[1].plot(q,r*q**2+u*q**4,ls,color='#4C72B0',label=lab)
axs[1].set_title(r'(b) $l=3$ (octupole): no cubic invariant',loc='left',fontsize=9.3)
axs[1].set_ylim(-0.003,0.012); axs[1].set_xlim(0,0.45)
for ax in axs:
    ax.set_xlabel(r'order-parameter amplitude $|\mathbf{q}^{(l)}|$'); ax.set_ylabel(r'Landau $f(q)$')
    ax.axhline(0,color='0.6',lw=0.6); ax.legend(loc='upper left'); ax.set_yticks([])
fig.tight_layout(w_pad=1.5); fig.savefig('fig/fig_landau.pdf')
