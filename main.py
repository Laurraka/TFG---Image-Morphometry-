#!/usr/bin/env python
# coding: utf-8

# In[1]:


import cv2
import random
import numpy as np
from scipy.spatial import Voronoi, cKDTree, voronoi_plot_2d
import matplotlib.pyplot as plt
import os
from scipy.optimize import linprog
import math
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


# In[2]:


# Masses per a cada partícula
def masses_voronoi(indexos, pixels, foto, N, height):
    masses=[0 for i in range(N)]
    for (punt,idx) in zip(pixels, indexos):
        masses[idx]+=foto[height-1-punt[1],punt[0]]
    return masses 


# In[3]:


# Cota superior d'error segons Observació 3.4
def upper_bond_error(particules, indexos, pixels, foto, N, height):
    sum=0
    for i in range(N):
        for (punt, idx) in zip(pixels, indexos):
            if idx==i:
                sum+=foto[(height-1)-punt[1], punt[0]]*np.linalg.norm(punt-particules[i])**2

    return sum


# In[4]:


def particle_approximation(foto, pixels, width, height, N, threshold_error=0.5, max_iteracions=1000):
    # Iniciem distribució de partícules aleatòria
    particules = np.random.randint(0, [width, height], size=(N, 2))

    # Inicialització
    iteracio = 0
    convergit = False

    while not convergit and iteracio < max_iteracions:
        iteracio += 1
        
        # Calculem diagrama de Voronoi per partícules i adjudiquem cada píxel a una cel·la
        vor = Voronoi(particules)
        kdtree = cKDTree(vor.points)
        distancia, indexos = kdtree.query(pixels)

        # Calculem masses de la mesura amb àtoms x_i
        mass=masses_voronoi(indexos, pixels, foto, N, height)
        
        #Esborrem les partícules que tinguin massa 0
        massa_0=[]
        for i in range(N):
            if mass[i]==0:
                massa_0.append(i)
                
        if len(massa_0)!=0:
            particules=np.delete(particules, np.sort(massa_0), axis=0)
            N=N-len(massa_0)
            #Tornem a crear diagrama de Voronoi amb les noves partícules
            vor = Voronoi(particules)
            kdtree = cKDTree(vor.points)
            distancia, indexos = kdtree.query(pixels)
            mass=masses_voronoi(indexos, pixels, foto, N, height)        
        
        error_anterior = upper_bond_error(particules, indexos, pixels, foto, N, height)

        # Recentrem els àtoms x_i
        particules_cm=[]
        for i in range(N):
            x=0
            for (punt, idx) in zip(pixels, indexos):
                if i==idx:
                    x+=foto[(height-1)-punt[1], punt[0]]*punt
            x=x/mass[i]
            particules_cm.append(x)
        
        particules_cm=np.array(particules_cm)

        # Calculem l'error actual
        error_actual=upper_bond_error(particules_cm, indexos, pixels, foto, N, height)

        # Calculem la diferència relativa d'error
        diferencia_error = abs(error_actual - error_anterior)
        error_relatiu = (diferencia_error / abs(error_actual)) * 100
        
        if error_relatiu < threshold_error:
            convergit = True
        else:
            particules=particules_cm
        
    return particules, particules_cm, N, error_actual


# In[5]:


def error_regions(particules, particules_cm, pixels, foto, N, height):
    vor = Voronoi(particules)
    kdtree = cKDTree(vor.points)
    distancia, indexos = kdtree.query(pixels)

    errors=[]
    for i in range(N):
        error_q=0
        for (punt, idx) in zip(pixels, indexos):
            if i==idx:
                error_q+=foto[height-1-punt[1], punt[0]]*(np.linalg.norm(punt-particules_cm[i]))**2
        errors.append(error_q)

    errors=[err**(1/2) for err in errors]

    return errors


# In[6]:


# Plotejar centres de massa en la imatge
def plot_punts_imatge(punts, foto, height):
    plt.figure(figsize=(6.4, 4.8))

    plt.imshow(foto, cmap='gray', origin='upper')
    
    for p in punts:
        plt.scatter(p[0], height-p[1], c='red', marker='o', s=5, alpha=0.7)
    
    plt.title("Centres de massa")
    plt.axis('off')
    plt.show()


# In[7]:


# Funció per a aplicar el pas 6
def afegir_particules(particules, particules_cm, pixels, foto, N, height, max_iteracions=20):
    # Iniciem: Mirem errors regions per mesura introduïda
    errors=error_regions(particules, particules_cm, pixels, foto, N, height)
    mitjana_errors=np.mean(errors)
    error_max=np.max(errors)

    iteracio=0

    while error_max>1.7*mitjana_errors and iteracio<max_iteracions:
        iteracio+=1
        idx_error_max=np.argmax(errors)

        vor=Voronoi(particules)
        kdtree = cKDTree(vor.points)
        distancia, indexos = kdtree.query(pixels)
    
        pixels_filtrats=[pixel for pixel, idx in zip(pixels, indexos) if idx==idx_error_max and foto[height-1-pixel[1], pixel[0]]!=0]
            
        seleccionats=random.sample(pixels_filtrats, 2)
        particules=np.delete(particules, idx_error_max, axis=0)
        particules= np.vstack([particules, seleccionats[0], seleccionats[1]])
    
        N+=1
    
        # Repetim el pas 2
        vor = Voronoi(particules)
        kdtree = cKDTree(vor.points)
        distancia, indexos = kdtree.query(pixels)
    
        # Calculem masses de la mesura amb àtoms x_i
        mass=masses_voronoi(indexos, pixels, foto, N, height)

        # Repetim el pas 3
        particules_cm=[]
        for i in range(N):
            x=0
            for (punt, idx) in zip(pixels, indexos):
                if i==idx:
                    x+=foto[height-1-punt[1], punt[0]]*punt
            x=x/mass[i]
            particules_cm.append(x)
            
        particules_cm=np.array(particules_cm)
    
        errors=error_regions(particules, particules_cm, pixels, foto, N, height)
        mitjana_errors=np.mean(errors)
        error_max=np.max(errors)

        if error_max>1.7*mitjana_errors:
            particules=particules_cm

    return particules, particules_cm, N


# ## Creació d'objecte Imatge

# In[8]:


class Imatge:
    def __init__(self, foto):
        # Preparem la imatge (normalitzar i intensitat igual a 1)
        foto_norm = cv2.normalize(foto, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
        self.foto_original = foto_norm
        foto_float = foto_norm.astype(np.float64)
        total = np.sum(foto_float)
        foto_I1=foto_float/total
        self.foto=foto_I1

        self.height, self.width = self.foto.shape
        self.N=300
        #self.N=(self.height*self.width)//300

        #Creem els píxels de la imatge
        self.pixels = []
        for x in range(self.width):
            for y in range(self.height):
                self.pixels.append([x, y])    
        self.pixels = np.array(self.pixels)

        voronoi, particules, self.N, error = particle_approximation(self.foto, self.pixels, self.width, self.height, self.N)
        self.voronoi, self.particules, self.N = afegir_particules(voronoi, particules, self.pixels, self.foto, self.N, self.height)

        vor = Voronoi(self.voronoi)
        kdtree = cKDTree(vor.points)
        distancia, indexos = kdtree.query(self.pixels)

        self.masses=masses_voronoi(indexos, self.pixels, self.foto, self.N, self.height)

    def mostrar_imatge(self):
        cv2.imshow('Imatge', self.foto_original) 
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def plot_particules(self):
        plot_punts_imatge(self.particules, self.foto_original, self.height)

    def plan(self, mesura):
        c=[]
        for i in range(self.N):
            for j in range(mesura.N):
                d=(np.linalg.norm(self.particules[i]-mesura.particules[j]))**2
                c.append(d)

        a=np.zeros((self.N + mesura.N, self.N*mesura.N))
        for i in range(self.N + mesura.N):
            for j in range(self.N*mesura.N):
                if i<self.N:
                    inici = i * mesura.N
                    fi = (i + 1) * mesura.N
                    a[i, inici:fi] = 1
                else:
                    if (j % mesura.N) == i-self.N:
                        a[i][j]=1
                    else:
                        a[i][j]=0

        b = np.concatenate([np.array(self.masses), np.array(mesura.masses)])

        plan = linprog(c, A_eq=a, b_eq=b)
        
        return plan.x

    def plan_sigma(self):
        self.f=self.plan(sigma)

    def distancia_wasser(self, mesura):
        c=[]
        for i in range(self.N):
            for j in range(mesura.N):
                d=(np.linalg.norm(self.particules[i]-mesura.particules[j]))**2
                c.append(d)
                
        return (c @ self.plan(mesura))**(1/2)

    def distancia_aLOT(self, sigma, mesura):
        f=self.f
        g=mesura.f

        y=[]
        z=[]
        
        for k in range(sigma.N):
            sum=0
            for i in range(self.N):
                sum+=f[k*self.N+i]*self.particules[i]
            
            sum=sum/sigma.masses[k]
            y.append(sum)
        
        for k in range(sigma.N):
            sum=0
            for j in range(mesura.N):
                sum+=g[k*mesura.N+j]*mesura.particules[j]
            
            sum=sum/sigma.masses[k]
            z.append(sum)

        dist=0
        for k in range(sigma.N):
            dist+=sigma.masses[k]*(np.linalg.norm(y[k]-z[k]))**2

        return dist**(1/2)


# In[9]:


# Fem que totes les imatges tinguin aproximadament les mateixes dimensions
carpeta_normal = 'dataset/color/normal'
carpeta_abnormal = 'dataset/color/abnormal'

M=20 #Nombre d'imatges de cada classe

alçades=[]
for fitxer in os.listdir(carpeta_normal)[:M]:
    ruta_completa = os.path.join(carpeta_normal, fitxer)
    foto = cv2.imread(ruta_completa)
    height = foto.shape[:2][0]
    alçades.append(height)

for fitxer in os.listdir(carpeta_abnormal)[:M]:
    ruta_completa = os.path.join(carpeta_abnormal, fitxer)
    foto = cv2.imread(ruta_completa)
    height = foto.shape[:2][0]
    alçades.append(height)

nova_alçada=np.min(alçades)


# In[10]:


celules_normals=[]

for fitxer in os.listdir(carpeta_normal)[:M]:
    ruta_completa = os.path.join(carpeta_normal, fitxer)
    foto = cv2.imread(ruta_completa, cv2.IMREAD_GRAYSCALE)
    height, width = foto.shape[:2]

    ratio=nova_alçada/height
    nova_amplada=int(width*ratio)

    foto_redim=cv2.resize(foto, (nova_amplada, nova_alçada))
    
    celula = Imatge(foto_redim)
    celules_normals.append(celula)


# In[11]:


celules_abnormals=[]

for fitxer in os.listdir(carpeta_abnormal)[:M]:
    ruta_completa = os.path.join(carpeta_abnormal, fitxer)
    foto = cv2.imread(ruta_completa, cv2.IMREAD_GRAYSCALE)
    height, width = foto.shape[:2]

    ratio=nova_alçada/height
    nova_amplada=int(width*ratio)

    foto_redim=cv2.resize(foto, (nova_amplada, nova_alçada))
    
    celula = Imatge(foto_redim)
    celules_abnormals.append(celula)


# In[12]:


# Computem la imatge "mitja"
imatges_redimensionades=[]

amplades=[]
for fitxer in os.listdir(carpeta_normal)[:M]:
    ruta_completa = os.path.join(carpeta_normal, fitxer)
    foto = cv2.imread(ruta_completa)
    width = foto.shape[:2][1]
    amplades.append(width)

for fitxer in os.listdir(carpeta_abnormal)[:M]:
    ruta_completa = os.path.join(carpeta_abnormal, fitxer)
    foto = cv2.imread(ruta_completa)
    width = foto.shape[:2][1]
    amplades.append(width)

nova_amplada=np.min(amplades)

for fitxer in os.listdir(carpeta_normal)[:M]:
    ruta_completa = os.path.join(carpeta_normal, fitxer)
    foto = cv2.imread(ruta_completa, cv2.IMREAD_GRAYSCALE)

    foto_redim=cv2.resize(foto, (nova_amplada, nova_alçada))
    imatges_redimensionades.append(foto_redim)

for fitxer in os.listdir(carpeta_abnormal)[:M]:
    ruta_completa = os.path.join(carpeta_abnormal, fitxer)
    foto = cv2.imread(ruta_completa, cv2.IMREAD_GRAYSCALE)

    foto_redim=cv2.resize(foto, (nova_amplada, nova_alçada))
    imatges_redimensionades.append(foto_redim)

stack_imatges = np.array(imatges_redimensionades, dtype=np.float64)
    
mitjana = np.mean(stack_imatges, axis=0).astype(np.uint8)

#cv2.imshow('Imatge', mitjana) 
#cv2.waitKey(0)
#cv2.destroyAllWindows()

sigma=Imatge(mitjana)


# In[13]:


celules=celules_normals+celules_abnormals


# In[38]:


celules[17].plot_particules()


# In[30]:


Y=[0 for i in range(M)]+[1 for i in range(M)]


# In[31]:


for I in celules:
    I.plan_sigma()


# In[32]:


# Computem distàncies Wasserstein dos a dos
dW=np.zeros((len(celules),len(celules)))
for i in range(len(celules)):
    for j in range(len(celules)):
        dW[i][j]=celules[i].distancia_wasser(celules[j])

print(dW)


# In[33]:


# Computem distàncies aLOT dos a dos
daLOT=np.zeros((len(celules),len(celules)))
for i in range(len(celules)):
    for j in range(len(celules)):
        daLOT[i][j]=celules[i].distancia_aLOT(sigma, celules[j])

print(daLOT)


# In[34]:


error=[]
for i in range(len(celules)):
    for j in range(len(celules)):
        if i<j:
            e=abs(dW[i][j]-daLOT[i][j])/dW[i][j]
            error.append(e)

print("Error mitjà:", np.mean(error))


# ## Support Vector Machine

# In[35]:


def exp_kernel(X, Y):
    gamma=2
    K = np.zeros((len(X), len(Y)))
    
    for i, x in enumerate(X):
        for j, y in enumerate(Y):
            K[i, j] = math.exp(-gamma*(x.distancia_wasser(y))**2)
    
    return K


# In[36]:


X_train, X_test, Y_train, Y_test = train_test_split(celules, Y, test_size=0.3)


# In[37]:


clf = svm.SVC(kernel=exp_kernel)
clf.fit(X_train, Y_train)


# In[38]:


Y_pred = clf.predict(X_test)
accuracy = accuracy_score(Y_test, Y_pred)
print(f"Accuracy: {accuracy:.4f}")


# # Fischer Linear Discriminant Analysis

# ## Immersió Euclidiana

# In[39]:


D=dW**2


# In[40]:


u=1/math.sqrt(2*M)*np.ones(2*M)


# In[41]:


Id=np.eye(2*M)


# In[42]:


G=-0.5*(Id-u@u.T)@D@(Id-u@u.T)


# In[43]:


vaps, veps = np.linalg.eig(G)


# In[44]:


idx = np.argsort(vaps)[::-1] 
vaps_ordenats = vaps[idx]
veps_ordenats = veps[:, idx]


# In[45]:


n_vaps_positius=sum(vaps>=0)


# In[46]:


correlacio_anterior=-1
correlacio=0
v=np.zeros(n_vaps_positius)
d=n_vaps_positius

while correlacio>correlacio_anterior:
    correlacio_anterior=correlacio
    v_ant=v

    arrels = np.sqrt(vaps_ordenats[:d])
    S = np.diag(arrels)

    U=np.array(veps_ordenats[:d])
    V=S @ U
    v=[V[:, i] for i in range(2*M)]

    DD=np.zeros((2*M,2*M))
    for i in range(2*M):
        for j in range(2*M):
            DD[i][j]=np.linalg.norm(v[i]-v[j])

    flat_a = D.flatten()
    flat_b = DD.flatten()
    correlacio = np.corrcoef(flat_a, flat_b)[0, 1]

    d=d-1

d+=2


# In[47]:


m_1=0
for i in range(M):
    m_1+=v_ant[i]

m_1=m_1/M

m_2=0 
for i in range(M,2*M):
    m_2+=v_ant[i]

m_2=m_2/M


# In[51]:


S_W=np.zeros((d,d))
for i in range(M):
    S_W+=(v_ant[i]-m_1).reshape(-1, 1)@(v_ant[i]-m_1).reshape(1, -1)

for i in range(M,2*M):
    S_W+=(v_ant[i]-m_2).reshape(-1, 1)@(v_ant[i]-m_2).reshape(1, -1)


# In[52]:


b=np.linalg.inv(S_W)@(m_2-m_1)
b=b/np.linalg.norm(b)


# In[53]:


def FLDA(x):
    return b@x


# In[54]:


k_min=np.argmin([FLDA(x) for x in v_ant])
k_max=np.argmax([FLDA(x) for x in v_ant])


# In[55]:


k_min, k_max


# In[56]:


celules[k_min].mostrar_imatge()


# In[57]:


celules[k_max].mostrar_imatge()


# In[58]:


def distancia_geodesica(I_0, I_1, I, alpha):
    # Computem I_alpha
    f=I_0.plan(I_1)

    particules_alpha=[]
    for i in range(I_0.N):
        for j in range(I_1.N):
            z=(1-alpha)*I_0.particules[i]+alpha*I_1.particules[j]
            particules_alpha.append(z)
    
    # Calculem d_OT(I_alpha, I)
    c=[]
    for i in range(I_0.N*I_1.N):
        for j in range(I.N):
            d=(np.linalg.norm(particules_alpha[i]-I.particules[j]))**2
            c.append(d)

    a=np.zeros((I_0.N*I_1.N + I.N, I_0.N*I_1.N*I.N))
    for i in range(I_0.N*I_1.N + I.N):
        for j in range(I_0.N*I_1.N*I.N):
            if i<I_0.N*I_1.N:
                inici = i * I.N
                fi = (i + 1) * I.N
                a[i, inici:fi] = 1
            else:
                if (j % I.N) == i-I_0.N*I_1.N:
                    a[i][j]=1
                else:
                    a[i][j]=0

    b = np.concatenate([np.array(f.x), np.array(I.masses)])

    plan = linprog(c, A_eq=a, b_eq=b)

    return (c @ plan.x)**(1/2)


# In[ ]:


indexos=[]
for alpha in np.arange(0.1, 1, 0.1):
    idx=np.argmin([distancia_geodesica(celules[k_min], celules[k_max], cel, alpha) for cel in celules])
    indexos.append(idx)


# In[ ]:




