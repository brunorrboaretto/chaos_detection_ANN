import colorednoise as cn
import numpy as np
import sys



#Serie = np.loadtxt(sys.argv[1]) 
#Size = len(Serie)
wl = 6
num_states = np.math.factorial(wl)

w1 = np.loadtxt('W1.dat')
W2 = np.loadtxt('W2.dat')
w2 = np.zeros((len(W2),1))
for i in range(len(W2)):
    w2[i] = W2[i]
b1 = np.loadtxt('B1.dat')
b2 = np.loadtxt('B2.dat')

def perm_indices(ts, wl, lag):
	m = len(ts)-(wl-1)*lag
	indcs = np.zeros(m, dtype=int)
	for i in range(1, wl):
		st = ts[(i-1)*lag: m + ((i-1)*lag)]
		for j in range(i, wl):
			indcs += st > ts[j*lag: m+j*lag]
		indcs *= wl-i
	return indcs + 1

def est_alpha (Serie,w1,w2,b1,b2):
  a1 = perm_indices(Serie, wl, 1)
  hist = np.histogram(a1,num_states)
  P = 1.0*hist[0]/len(a1)
  P_data = np.array([P])
  S = 0.0
  for i in range(num_states):
    if P[i]!= 0.0:
      S-=(P[i]*np.log(P[i]))
  return [np.add(np.dot(np.maximum(np.add(np.dot(P_data[0],w1),b1),0.0),w2),b2),S]

def evaluate_omega(Serie,num_cis):
  Size= len(Serie)
  alpha, S,= est_alpha(Serie,w1,w2,b1,b2)
  S_aux = np.zeros(num_cis)

  for i in range(num_cis):
    np.random.seed(i)
    Serie_aux = cn.powerlaw_psd_gaussian(float(alpha), Size)
    _, S_aux[i] = est_alpha(Serie_aux,w1,w2,b1,b2)
  S/=np.log(num_states)
  S_aux/=np.log(num_states)
  D = np.zeros(num_cis)
  for i in range(num_cis):
    D[i]=abs(S - S_aux[i])/S_aux[i]
  return (float (alpha),S,np.mean(S_aux),np.mean(D))

#print(evaluate_omega(Serie,10))