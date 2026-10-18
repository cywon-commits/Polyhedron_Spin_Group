from style import *; import numpy as np
fig,ax=plt.subplots(figsize=(COL,2.7))
lab=[r'$C_3^{+}$',r'$C_2$',r'$C_3^{-}$',r'$-1$']
th=np.array([2/3,1,4/3,2]); E=th**2
ax.bar(range(4),E,0.55,color=['#4C72B0','#55A868','#DD8452','#C44E52'])
for i,(t,e) in enumerate(zip(th,E)):
    ax.text(i,e+0.12,r'$\theta=%s\pi$'%({2/3:'2/3',1:'',4/3:'4/3',2:'2'}[t]) ,ha='center',fontsize=8.5)
ax.set_xticks(range(4)); ax.set_xticklabels(lab)
ax.set_ylabel(r'line energy $\propto\theta^2$ ($\pi^2$ units)')
ax.set_ylim(0,5.0)
ax.text(-0.42,4.75,'splitting (energy $\\propto\\sum\\theta^2$):\n'+r'$-1\to C_2C_2$:  $4\to2$'+'\n'+r'$C_3^-\to(C_3^+)^2$:  $16/9\to8/9$'+'\n'+r'$C_2\to C_3^+C_3^+$:  $1\to8/9$',va='top',fontsize=8.5)
ax.set_xlabel(r'conjugacy class of $2T$')
fig.savefig('fig/fig_defects.pdf')
