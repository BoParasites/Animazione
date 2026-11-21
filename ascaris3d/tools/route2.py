import json, numpy as np
from scipy.spatial import cKDTree
def load(n):
    d=json.load(open(f'mesh/{n}.json')); return np.array(d['V']).reshape(-1,3), np.array(d['F']).reshape(-1,3)
cl={k:np.array(v) for k,v in json.load(open('centerlines.json')).items()}
M={n:load(n)[0] for n in ['stomach','small_intestine','colon','liver','veins_abdomen','lung_r','heart','bronchi','skin','vena_cava_inferior','pulmonary_vessels']}
T={n:cKDTree(v) for n,v in M.items()}
def medial(P,n,r):
    V=M[n]; out=[]
    for p in P:
        idx=T[n].query_ball_point(p,r)
        out.append(V[idx].mean(0) if len(idx)>5 else np.array(p,float))
    return np.array(out)
def smooth(P,it=2):
    P=np.array(P,float)
    for _ in range(it): P[1:-1]=0.25*P[:-2]+0.5*P[1:-1]+0.25*P[2:]
    return P
def resample(P,step):
    P=np.array(P,float); d=np.r_[0,np.cumsum(np.linalg.norm(np.diff(P,axis=0),axis=1))]
    s=np.arange(0,d[-1],step); return np.array([np.interp(s,d,P[:,k]) for k in range(3)]).T

S={}
mouth=np.array([0,1.527,0.094])
S['ingest']=[mouth,[0,1.524,0.06],[0,1.515,0.025],[0,1.49,0.0],*cl['eso'][1:].tolist()]
# stomaco: cardias -> fondo -> corpo -> antro -> piloro
st_guess=[[0.03,1.205,0.02],[0.06,1.215,0.03],[0.08,1.19,0.045],[0.075,1.16,0.06],[0.055,1.14,0.07],[0.025,1.135,0.072],[0.0,1.138,0.065],[-0.022,1.142,0.056]]
st=medial(st_guess,'stomach',0.028)
S['ingest']+=smooth(st,1).tolist()
# tenue: anse a serpentina, ancorate al volume reale
SI=M['small_intestine']; rows=[]
ys=np.linspace(1.115,0.935,8)
for i,y in enumerate(ys):
    xs=np.linspace(-0.06,0.085,7)
    if i%2: xs=xs[::-1]
    for x in xs:
        z=np.median(SI[(abs(SI[:,1]-y)<0.02)&(abs(SI[:,0]-x)<0.03)][:,2]) if ((abs(SI[:,1]-y)<0.02)&(abs(SI[:,0]-x)<0.03)).sum()>3 else 0.05
        rows.append([x,y,z])
rows=medial(rows,'small_intestine',0.022)
# duodeno/primo digiuno: dal piloro ai primi punti
first=rows[:5]
S['hatch']=[[-0.03,1.13,0.05],[-0.035,1.115,0.045],*first[:3].tolist()]
# penetrazione -> vena mesenterica -> porta -> fegato
portal_guess=[[0.0,1.10,0.04],[0.002,1.115,0.025],[-0.01,1.13,0.02],[-0.025,1.15,0.025],[-0.04,1.165,0.03]]
portal=medial(portal_guess,'veins_abdomen',0.012)
liver=medial([[-0.06,1.175,0.03],[-0.08,1.19,0.02],[-0.075,1.215,0.0],[-0.05,1.235,-0.005],[-0.03,1.245,0.0]],'liver',0.03)
ivc_top=cl['ivc'][cl['ivc'][:,1]>1.2]
S['liver']=[*portal.tolist(),*liver.tolist(),*ivc_top.tolist()]
heart=[[-0.012,1.275,0.012],[-0.02,1.295,0.02],[-0.005,1.29,0.045],[0.015,1.30,0.06],[0.02,1.325,0.055],[0.012,1.348,0.035],[0.0,1.355,0.015],[-0.025,1.352,0.0]]
S['heart']=heart
lung=medial([[-0.05,1.34,-0.01],[-0.07,1.30,0.0],[-0.085,1.25,0.005],[-0.092,1.20,0.0]],'lung_r',0.03)
S['lungs']=lung.tolist()
br=M['bronchi']; brR=br[br[:,0]<-0.005]
asc=medial([[-0.088,1.215,-0.01],[-0.075,1.27,-0.015],[-0.05,1.32,-0.02],[-0.03,1.35,-0.025],[-0.012,1.365,-0.02]],'lung_r',0.02).tolist()
S['ascent']=asc+cl['tra'].tolist()+[[0,1.47,0.0],[0,1.50,0.005]]
eso2=(cl['eso']+np.array([0.006,0,0.004]))[1:]
st2=medial(st_guess,'stomach',0.035)+np.array([0.004,-0.004,0.0])
S['swallow']=[[0.003,1.49,-0.006],*eso2.tolist(),*smooth(st2,1).tolist(),[-0.03,1.13,0.052],*first[:2].tolist()]
S['adult']=rows[2:].tolist()
# colon: dal cieco al retto, ripulito
col=cl['col']; col=resample(smooth(col,3),0.012); col=medial(col,'colon',0.02); col=smooth(col,2)
S['eggs']=[*col.tolist(),[0,0.815,-0.055],[0,0.79,-0.045]]
S['soil']=[[0,0.74,-0.02],[0.0,0.55,0.03],[0.03,0.25,0.07],[0.08,0.06,0.16],[0.15,0.012,0.24],[0.24,0.008,0.27],[0.33,0.008,0.25],[0.40,0.008,0.20]]
S['contagion']=[[0.45,0.03,0.15],[0.40,0.30,0.12],[0.30,0.62,0.08],[0.272,0.74,0.07],[0.27,0.82,0.075],[0.26,0.95,0.12],[0.20,1.20,0.20],[0.11,1.42,0.20],[0.04,1.51,0.15]]
out={k:np.round(np.array(v,float),4).tolist() for k,v in S.items()}
json.dump(out,open('route3d.json','w'))
for k,v in out.items(): print(k,len(v))
