import numpy as np 
import matplotlib.pyplot as plt 
from NN_Omega import perm_indices,est_alpha,evaluate_omega 
import colorednoise as cn
from fbm import fbm, fgn, times

Size = int(2**16)
CI = 101

fig, ax = plt.subplots(2, 1, 
                           gridspec_kw={
                               #'width_ratios': [1, 1],
                               'height_ratios': [3, 1]})


print('---------- 0%')


#########################################################
#----------------------STOCHASTIC-----------------------#
#########################################################

#Flicker Noise

a = np.zeros(CI)
S = np.zeros(CI)
O = np.zeros(CI)
    
for seed in range(CI):    
    np.random.seed(seed)
    alpha = 4.0*seed/CI - 1.0#np.random.rand()*4.0 - 1.0
    Serie = cn.powerlaw_psd_gaussian(float(alpha), Size)
    a[seed],S[seed],_,O[seed] = evaluate_omega(Serie,1)
    
ax[0].plot(a,S,'k-',label='Flicker noise')#,markersize=10,mfc='w',mec='r',mew=2,label=r'$\beta x$ Map',alpha=0.8)
ax[1].plot(a,O,'k-')#,markersize=10,mfc='w',mec='r',mew=2,label=r'$\beta x$ Map',alpha=0.8)

#########################################################
print('>--------- 10%')

#fBm and fGn

a = np.zeros(CI)
S = np.zeros(CI)
O = np.zeros(CI)
    
for seed in range(CI):    
    np.random.seed(seed)
    H = 0.6*seed/CI + 0.2
    Serie = fbm(n=Size, hurst=H)
    a[seed],S[seed],_,O[seed] = evaluate_omega(Serie,1)
    
ax[0].plot(a,S,'o',mec='k',mfc='r',markersize=5,label=r'fBm',alpha=1)#,markersize=10,mfc='w',mec='r',mew=2,label=r'$\beta x$ Map',alpha=0.8)
ax[1].plot(a,O,'o',mec='k',mfc='r',markersize=5,alpha=1)#,markersize=10,mfc='w',mec='r',mew=2,label=r'$\beta x$ Map',alpha=0.8)
print('=>-------- 20%')


a = np.zeros(CI)
S = np.zeros(CI)
O = np.zeros(CI)
    
for seed in range(CI):    
    np.random.seed(seed)
    H = 0.6*seed/CI + 0.2
    Serie = fgn(n=Size, hurst=H)
    a[seed],S[seed],_,O[seed] = evaluate_omega(Serie,1)
    
ax[0].plot(a,S,'s',mec='k',mfc='g',markersize=5,label=r'fGn',alpha=1)#,markersize=10,mfc='w',mec='r',mew=2,label=r'$\beta x$ Map',alpha=0.8)
ax[1].plot(a,O,'s',mec='k',mfc='g',markersize=5,alpha=1)#,markersize=10,mfc='w',mec='r',mew=2,label=r'$\beta x$ Map',alpha=0.8)
print('==>------- 30%')


#Cauchy noise


a = np.zeros(CI)
S = np.zeros(CI)
O = np.zeros(CI)
    
for seed in range(CI):    
    np.random.seed(seed)
    Serie = np.random.standard_cauchy(Size);
    a[seed],S[seed],_,O[seed] = evaluate_omega(Serie,1)
    
ax[0].plot(np.mean(a),np.mean(S),'^',mec='k',mfc='b',markersize=10,label=r'Cauchy',alpha=1)#,markersize=10,mfc='w',mec='r',mew=2,label=r'$\beta x$ Map',alpha=0.8)
ax[1].plot(np.mean(a),np.mean(O),'^',mec='k',mfc='b',markersize=10,alpha=1)#,markersize=10,mfc='w',mec='r',mew=2,label=r'$\beta x$ Map',alpha=0.8)

print('===>------ 40%')


#########################################################

#Uniform Noise



a = np.zeros(CI)
S = np.zeros(CI)
O = np.zeros(CI)
    
for seed in range(CI):    
    np.random.seed(seed)
    Serie = np.random.uniform(0.0, 1.0,Size)
    a[seed],S[seed],_,O[seed] = evaluate_omega(Serie,1)
    
ax[0].plot(np.mean(a),np.mean(S),'s',mec='k',mfc='m',markersize=6,label=r'Uniform',alpha=1)#,markersize=10,mfc='w',mec='r',mew=2,label=r'$\beta x$ Map',alpha=0.8)
ax[1].plot(np.mean(a),np.mean(O),'s',mec='k',mfc='m',markersize=6,alpha=1)#,markersize=10,mfc='w',mec='r',mew=2,label=r'$\beta x$ Map',alpha=0.8)

print('====>----- 50%')


#########################################################
#---------------------DETERMINISTIC---------------------#
#########################################################

#Beta x map


a = np.zeros(CI)
S = np.zeros(CI)
O = np.zeros(CI)
    
for seed in range(CI):    
    np.random.seed(seed)
    x0 = np.random.rand()
    Serie = np.zeros(Size)
    for i in range(Size):
        x=2.00000001*x0
        while x>=1.0:
            x-=1.0
        x0=x 
        Serie[i] = x0
    
    a[seed],S[seed],_,O[seed] = evaluate_omega(Serie,1)
    
ax[0].plot(np.mean(a),np.mean(S),'ro',markersize=10,mfc='w',mec='r',mew=2,label=r'$\beta x$ Map',alpha=0.8)
ax[1].plot(np.mean(a),np.mean(O),'ro',markersize=10,mfc='w',mec='r',mew=2,label=r'$\beta x$ Map',alpha=0.8)
print('=====>---- 60%')

#########################################################

#Logistic map



a = np.zeros(CI)
S = np.zeros(CI)
O = np.zeros(CI)
    
for seed in range(CI):    
    np.random.seed(seed)
    x0 = np.random.rand()
    Serie = np.zeros(Size)
    for i in range(Size):
        x=4.0*x0*(1-x0)
        x0=x 
        Serie[i] = x0
    
    a[seed],S[seed],_,O[seed] = evaluate_omega(Serie,1)
    
ax[0].plot(np.mean(a),np.mean(S),'gs',markersize=10,mfc='w',mec='g',mew=2,label=r'Logistic Map',alpha=0.8)
ax[1].plot(np.mean(a),np.mean(O),'gs',markersize=10,mfc='w',mec='g',mew=2,label=r'Logistic Map',alpha=0.8)

print('======>--- 70%')


#########################################################

#Skew Tent Map

a = np.zeros(CI)
S = np.zeros(CI)
O = np.zeros(CI)
    
w = 0.1847
x0 = np.random.rand()

for seed in range(CI):    
    np.random.seed(seed)
    x0 = np.random.rand()
    Serie = np.zeros(Size)
    for i in range(Size):
        if x0 < w:
            x = x0/w
        if x0 > w:
            x = (1-x0)/(1-w)
        x0=x 
        Serie[i] = x0
    
    a[seed],S[seed],_,O[seed] = evaluate_omega(Serie,1)
    
ax[0].plot(np.mean(a),np.mean(S),'mD',markersize=10,mfc='w',mec='m',mew=2,label=r'Skew Tent Map',alpha=0.8)
ax[1].plot(np.mean(a),np.mean(O),'mD',markersize=10,mfc='w',mec='m',mew=2,label=r'Skew Tent Map',alpha=0.8)

print('=======>-- 80%')


#########################################################

#Schuster Map


a = np.zeros(CI)
S = np.zeros(CI)
O = np.zeros(CI)
    
w = 0.5
x0 = np.random.rand()

for seed in range(CI):    
    np.random.seed(seed)
    x0 = np.random.rand()
    Serie = np.zeros(Size)
    for i in range(Size):
        x = x0+x0**w
        while x>=1.0:
            x-=1.0
        x0=x 
        Serie[i] = x0
    
    a[seed],S[seed],_,O[seed] = evaluate_omega(Serie,1)
    
ax[0].plot(np.mean(a),np.mean(S),'b^',markersize=10,mfc='w',mec='b',mew=2,label=r'Schuster Map',alpha=0.8)
ax[1].plot(np.mean(a),np.mean(O),'b^',markersize=10,mfc='w',mec='b',mew=2,label=r'Schuster Map',alpha=0.8)

print('========>- 90%')


#########################################################


sz1 = 23
sz2 = 18
ax[0].set_xlim([-1,3])
ax[0].set_ylim([-0.05,1.05])
ax[0].set_xticks(np.linspace(-1,3,5))
ax[0].set_xticklabels((' ',' ',' ',' ',' '))
ax[0].set_yticks(np.linspace(0,1,5))
ax[0].set_yticklabels( ('$0.00$','$0.25$','$0.50$','$0.75$','$1.00$'))  
ax[0].tick_params(axis='both', labelsize=sz2)
ax[0].legend(numpoints=1,loc='lower right',ncol=2,fontsize=10,framealpha=0.8)
ax[0].set_ylabel(r'$\bar{S}(\alpha_\mathrm{e})$', fontsize=sz1)
ax[1].set_xlim([-1,3])
ax[1].set_ylim([-0.05,1.05])
ax[1].set_xticks(np.linspace(-1,3,5))
ax[1].set_xticklabels(('$-1$','$0$','$1$','$2$','$3$'))
ax[1].set_yticks(np.linspace(0,1,5))
ax[1].set_yticklabels( ('$0.00$',' ','$0.50$',' ','$1.00$'))
ax[1].tick_params(axis='both', labelsize=sz2)

#ax[1].legend(numpoints=1,loc='lower right',ncol=2,fontsize=14,framealpha=0.8)
ax[1].set_xlabel(r'$\alpha_\mathrm{e}$', fontsize=sz1) 
ax[1].set_ylabel(r'$\Omega(\alpha_\mathrm{e})$', fontsize=sz1)

width = 28; height = 14;
fig.set_size_inches(width/2.54,height/2.54) #2.54 cm = 1 inches
#plt.subplots_adjust(left=None, bottom=.2, right=None, top=None, wspace=None, hspace=None)
plt.savefig('test_fig3.png', dpi=250)
print('=========> 100%')
print('========== Done')
print('test_fig3.png has been generated!)

#plt.show()

