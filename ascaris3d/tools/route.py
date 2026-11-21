import json, numpy as np
from scipy.cluster.vq import kmeans2
from scipy.sparse.csgraph import shortest_path, connected_components
from scipy.sparse import coo_matrix
np.random.seed(3)
def load(n):
    d=json.load(open(f'mesh/{n}.json')); return np.array(d['V']).reshape(-1,3), np.array(d['F']).reshape(-1,3)

def slice_centerline(V, axis=1, n=12, lo=None, hi=None):
    a=V[:,axis]; lo=a.min() if lo is None else lo; hi=a.max() if hi is None else hi
    out=[]
    for t in np.linspace(lo,hi,n+1)[:-1]:
        w=(hi-lo)/n; m=(a>=t)&(a<t+w)
        if m.sum()>3: out.append(V[m].mean(0))
    return np.array(out)

def cluster_graph(V,F,K):
    cent,lab=kmeans2(V,K,minit='++',seed=3)
    e=np.vstack([F[:,[0,1]],F[:,[1,2]],F[:,[2,0]]]); a=lab[e[:,0]]; b=lab[e[:,1]]; m=a!=b
    D=np.linalg.norm(cent[a[m]]-cent[b[m]],axis=1)
    G=coo_matrix((D,(a[m],b[m])),shape=(K,K)).tocsr(); G=G.maximum(G.T)
    return cent,lab,G

def medial(P,V,r):
    out=[]
    for p in P:
        d=np.linalg.norm(V-p,axis=1); m=d<r
        out.append(V[m].mean(0) if m.sum()>5 else p)
    return np.array(out)

def tube_path(V,F,K,start,end,r):
    cent,lab,G=cluster_graph(V,F,K)
    s=np.argmin(np.linalg.norm(cent-start,axis=1)); e=np.argmin(np.linalg.norm(cent-end,axis=1))
    D,P=shortest_path(G,directed=False,indices=s,return_predecessors=True)
    path=[e]
    while path[-1]!=s and P[path[-1]]>=0: path.append(P[path[-1]])
    pts=cent[path[::-1]]
    return medial(pts,V,r)

R={}
# bocca, faringe, esofago
E,EF=load('esophagus')
eso=slice_centerline(E,1,14)[::-1]   # dall'alto in basso
mouth=np.array([0,1.527,0.092]); lips_in=np.array([0,1.525,0.06]); oro=np.array([0,1.515,0.02]); pharynx=np.array([0,1.49,-0.005])
# stomaco
St,SF=load('stomach')
pyl=St[np.argmin(St[:,0]-0.3*St[:,2]+0.0*St[:,1])]  # più a destra (−x) e anteriore
stom=tube_path(St,SF,24,eso[-1],pyl,0.035)
R['stomach_pts']=stom
# tenue: meandro su grafo di cluster (visita in profondità del vicino non visitato più vicino)
Si,SiF=load('small_intestine')
cent,lab,G=cluster_graph(Si,SiF,70)
G=G.toarray()
start=np.argmin(np.linalg.norm(cent-pyl,axis=1))
order=[start]; vis={start}
while len(vis)<len(cent):
    c=order[-1]; nb=[j for j in np.nonzero(G[c])[0] if j not in vis]
    if not nb: nb=[j for j in range(len(cent)) if j not in vis]
    j=min(nb,key=lambda j:np.linalg.norm(cent[j]-cent[c])); order.append(j); vis.add(j)
si=medial(cent[order],Si,0.02)
# colon: dal cieco al retto
C,CF=load('colon')
cecum=C[np.argmin(C[:,1]+ (C[:,0]>-0.03)*1.0)]   # punto più basso del lato destro
rect=C[np.argmin(C[:,1])]
col=tube_path(C,CF,60,cecum,rect,0.03)
# cava inferiore
I,_=load('vena_cava_inferior'); ivc=slice_centerline(I,1,10)
T,_=load('trachea'); tra=slice_centerline(T,1,8)
B,BF=load('bronchi')
L,_=load('lung_r'); H,_=load('heart'); Lv,_=load('liver')
out={k:(v.tolist() if hasattr(v,'tolist') else v) for k,v in dict(eso=eso,stom=stom,si=si,col=col,ivc=ivc,tra=tra,mouth=mouth,pyl=pyl,cecum=cecum,rect=rect).items()}
json.dump(out,open('centerlines.json','w'))
for k,v in out.items(): print(k, np.round(np.array(v),3).tolist()[:4], '...', len(v) if isinstance(v[0],list) else '')
