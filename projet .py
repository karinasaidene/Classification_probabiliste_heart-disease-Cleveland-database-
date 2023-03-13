import math
from pandas._libs.tslibs.dtypes import PeriodDtypeBase # for math
import utils
import numpy as np
import pandas as pd
import scipy
import matplotlib.pyplot as plt

#========question 1=======
def getPrior(df) :
    """
    @param df : donnee
    @return : un dictionnaire à 3 clés 
                    -estimation
                    -min5pourcentage et max5pourcentage :intervalle de confiance à 95%
    """
    estimation = df['target'].mean()
    variance = df['target'].var()
    nb = df.shape[0]
    min5pourcent =  df['target'].mean() - 1.96 * math.sqrt(variance)/math.sqrt(nb)
    max5pourcent =  df['target'].mean() + 1.96 * math.sqrt(variance)/math.sqrt(nb)
    return {'estimation': estimation , 'min5pourcent' : min5pourcent , 'max5pourcent' : max5pourcent}

#===========question 2.a=====
class APrioriClassifier(utils.AbstractClassifier):
  """
  Classifieur à priori
  """
  def __init__(self,df):
    if getPrior(df)['estimation'] >= 0.5 :
      self.apioriClassifier = 1
    else :
      self.apioriClassifier = 0

  def estimClass(self, attrs):
    return self.apioriClassifier
#==========question 2.b========
  def statsOnDF(self, df):
    VP = 0
    VN = 0
    FP = 0
    FN = 0
    for i in range(df.shape[0]):
      dict = utils.getNthDict(df,i)
      if(self.estimClass(dict) == 1):
        if dict['target'] ==1 :
          VP +=1
        else :
          FP += 1
      else :
        if dict['target'] ==0 :
          VN +=1
        else :
          FN += 1
    precision = VP / (VP + FP)
    rappel = VP / (FN + VP)
    return {'VP': VP, 'VN': VN, 'FP': FP, 'FN': FN, 'Précision': precision,'Rappel': rappel}

#======question 3.a ============= 
def P2D(df, attr1, attr2):
  """
  calcule la probabbilité de l'attribut sachant la valeur de target
  @param df :donne
  @param attr : attribut
  @return  : un dictionnaire de probabilité
  """
  attr1_values = np.unique(np.array(df[attr1]))
  attr2_values = np.unique(np.array(df[attr2]))
  proba = dict()
  i = 1
  stop1 = False
  while(i!=1 or (not stop1)):
    p = dict()
    j = 1
    stop2 = False
    while(j!=1 or (not stop2 )):
      p[attr2_values[j]] =  df[(df[attr1] ==attr1_values[i]) & (df[attr2] == attr2_values[j])].shape[0] / df[df[attr1] ==attr1_values[i]].shape[0]
      if j == len(attr2_values)-1 : 
        j = 0
      else :
        j+=1
      stop2 = True
    proba[attr1_values[i]] = p
    if i == len(attr1_values)-1 : 
        i = 0
    else :
        i+=1
    stop1 = True
  return proba

def P2D_l(df , attr):
  return P2D(df,'target',attr)

def P2D_p(df , attr) :
  """
  calcule la probabbilité de target sachant la valeur d'un attribut
  @param df :donne
  @param  attr : attribut
  @return  : un dictionnaire de probabilité
  """
  return P2D(df,attr,'target')

#===========question 3.b=========== 
class ML2DClassifier(APrioriClassifier) :
  """
  Classifieur 2D maximum de vraissemblance
  """
  def __init__ (self,df, attr) :
    self.attr = attr
    self.P2Dl = P2D_l(df, attr)

 
  def estimClass (self, attrs):
    attribut = attrs [self.attr]
    attr_target0 = self.P2Dl[0][attribut]
    attr_target1 = self.P2Dl[1][attribut]

    if attr_target0 > attr_target1 :
      return 0
    else :
      return 1
#===========question 3.C==============
class MAP2DClassifier(APrioriClassifier) :
  """
  Classifieur 2D maximum à posteriori
  """
  def __init__(self,df , attr) :
    self.attr = attr
    self.P2Dp = P2D_p(df, attr)

 
  def estimClass (self, attrs):
    attribut = attrs [self.attr]
    attr_target0 = self.P2Dp[attribut][0]
    attr_target1 = self.P2Dp[attribut][1]

    if attr_target0 > attr_target1 :
      return 0
    else :
      return 1

#================question 4.1===========
def nbParams(df, attrs=None) :
  """
  @param df : données
  @param attrs: liste d'attributs à prendre en compte 
  @return :retourne le nombre d'octets en mémoire nécessaire pour la construction d'un x-classifieur
  """
  cpt =1
  if attrs == None :
    attrs=[str(key) for key in df.keys()]
  for attr in attrs :
    dic = np.unique(np.array(df[attr]))
    cpt *= len(dic)
  nb_octet = cpt * 8 #car un float est sur 8 octets

  taille_o = 0
  taille_ko = 0
  taille_mo = 0
  taille_go = 0
  if nb_octet>=1024 :
    taille_o = nb_octet%1024
    taille_ko = nb_octet//1024
    if taille_ko >= 1024 :
      taille_mo = taille_ko//1024
      taille_ko = taille_ko %1024
      if taille_mo >=1024:
        taille_go = taille_mo //1024
        taille_mo = taille_mo % 1024
        print (str(len(attrs)) +" variable(s) : "+ str(nb_octet)+" octets = "+ str(taille_go) +"go "+str(taille_mo)+"mo "+ str(taille_ko)+"ko "+ str(taille_o) +"o")
      else : 
        print (str(len(attrs)) +" variable(s) : "+ str(nb_octet)+" octets = "+ str(taille_mo) +"mo "+ str(taille_ko) +"ko "+ str(taille_o)+"o")
    else :
      print (str(len(attrs)) +" variable(s) : "+ str(nb_octet)+" octets = "+ str(taille_ko) +"ko "+ str(taille_o)+"o")
  else : 
    print (str(len(attrs)) +" variable(s) : "+ str(nb_octet)+" octets")
  
#========question 4.2=====================
def nbParamsIndep(df , attrs =None) :
  """
  @param  df : donnee
  @param  attrs : liste des attributs à prendre en compte
  @return  : le nombre d'octets en mémoire nécessaire pour la construction d'un x-classifieur dont
            on ne stocke que les produits des probabilités
  """
  cpt =0
  if attrs == None :
    attrs=[str(key) for key in df.keys()]
  for attr in attrs :
    dic = np.unique(np.array(df[attr]))
    cpt += len(dic)
  nb_octet = cpt * 8 #car un float est sur 8 octets

  taille_o = 0
  taille_ko = 0
  taille_mo = 0
  taille_go = 0
  if nb_octet>=1024 :
    taille_o = nb_octet%1024
    taille_ko = nb_octet//1024
    if taille_ko >= 1024 :
      taille_mo = taille_ko//1024
      taille_ko = taille_ko %1024
      if taille_mo >=1024:
        taille_go = taille_mo //1024
        taille_mo = taille_mo % 1024
        print (str(len(attrs)) +" variable(s) : "+ str(nb_octet)+" octets = "+ str(taille_go) +"go "+str(taille_mo)+"mo "+ str(taille_ko)+"ko "+ str(taille_o) +"o")
      else : 
        print (str(len(attrs)) +" variable(s) : "+ str(nb_octet)+" octets = "+ str(taille_mo) +"mo "+ str(taille_ko) +"ko "+ str(taille_o)+"o")
    else :
      print (str(len(attrs)) +" variable(s) : "+ str(nb_octet)+" octets = "+ str(taille_ko) +"ko "+ str(taille_o)+"o")
  else : 
    print (str(len(attrs)) +" variable(s) : "+ str(nb_octet)+" octets")
#=========question 5.3=============  
def drawNaiveBayes(df,root):
  """
  prend un dessin de l'attribut en fonction du reste des attributs du dataframe
  """
  args = ""
  attrs = df.keys()
  for attr in attrs :
    if(root != attr):
      args += root + "->" +  attr + ";"
  return utils.drawGraph(args)

def nbParamsNaiveBayes(df, attr , attrs=None) :
  """
  @param  df : donnee
  @param  attr : l'attribut en fonction duquel le calcul est fait
  @param  attrs : liste d'attributs à prendre en compte
  @return  :  le nombre d'octet memoire pris par naive bayes
  """
  size = 0
  if attrs is None :
    attrs = df.keys()
  for a in attrs :
    if(a!=attr):
      size += len(np.unique(df[a]))
  nb_attr = len(np.unique(df[attr]))
  size *= nb_attr
  size += nb_attr
  size *= 8
  
  taille_o = 0
  taille_ko = 0
  taille_mo = 0
  taille_go = 0
  if size>=1024 :
    taille_o = size%1024
    taille_ko = size//1024
    if taille_ko >= 1024 :
      taille_mo = taille_ko//1024
      taille_ko = taille_ko %1024
      if taille_mo >=1024:
        taille_go = taille_mo //1024
        taille_mo = taille_mo % 1024
        print (str(len(attrs)) +" variable(s) : "+ str(size)+" octets = "+ str(taille_go) +"go "+str(taille_mo)+"mo "+ str(taille_ko)+"ko "+ str(taille_o) +"o")
      else : 
        print (str(len(attrs)) +" variable(s) : "+ str(size)+" octets = "+ str(taille_mo) +"mo "+ str(taille_ko) +"ko "+ str(taille_o)+"o")
    else :
      print (str(len(attrs)) +" variable(s) : "+ str(size)+" octets = "+ str(taille_ko) +"ko "+ str(taille_o)+"o")
  else : 
    print (str(len(attrs)) +" variable(s) : "+ str(size)+" octets")

#==============question 5.4============
class MLNaiveBayesClassifier(APrioriClassifier):
    """
    Classifieur maximum de vraissemblance 
    """
    def __init__(self, df):
      self.P2DL_attrs = dict()
      for attr in df.keys():
        self.P2DL_attrs[attr] = P2D_l(df,attr)
      self.probas = {0: None, 1: None}
    
    def estimProbas(self,attrs):
      target_0 = 1
      target_1 = 1
      for attr in attrs.keys():
        if attrs[attr] not in self.P2DL_attrs[attr][0].keys():
          self.probas[0] = 0
          self.probas[1] = 0
          return self.probas
        if attr != "target":
          target_0 *= self.P2DL_attrs[attr][0][attrs[attr]]
          target_1 *= self.P2DL_attrs[attr][1][attrs[attr]]
      self.probas[0] = target_0
      self.probas[1] = target_1
      return self.probas

    def estimClass(self, attrs):
        self.probas = self.estimProbas(attrs)
        if self.probas[0] >= self.probas[1]:
            return 0
        return 1

class MAPNaiveBayesClassifier(APrioriClassifier):
  """
  Classifieur maximum à posteriori
  """
  def __init__(self, df):
    self.P2DL_attrs = dict()
    for attr in df.keys():
      self.P2DL_attrs[attr] = P2D_l(df,attr)
    self.p_t0 = df[(df['target'] == 0)].shape[0]/len(df) #P(target=0)
    self.p_t1 = df[(df['target'] == 1)].shape[0]/len(df) #P(target=1)
      

  def estimProbas(self,attrs):
      probas = {0: None, 1: None}
      target_0 = 1
      target_1 = 1
      for attr in attrs.keys():
        if attrs[attr] not in self.P2DL_attrs[attr][0].keys():
          probas[0] = 0
          probas[1] = 0
          return probas
        if attr != "target":
          target_0 *= self.P2DL_attrs[attr][0][attrs[attr]]
          target_1 *= self.P2DL_attrs[attr][1][attrs[attr]]
      somme = self.p_t0 *  target_0 + self.p_t1* target_1
      if somme == 0:
        probas[0] = 0
        probas[1] = 0
        return probas
      probas[0] = target_0*self.p_t0/somme
      probas[1] = target_1*self.p_t1/somme
      return probas
  def estimClass(self, attrs):
      probas = self.estimProbas(attrs)
      if probas[0] >= probas[1]:
          return 0
      return 1
#==================Question 6===========
def isIndepFromTarget(df , attr , x) :
  """
  @param  df : donnes 
  @param  attr : attribut
  @param  x  parametre alpha
  @return  : verifie si attr est independante de target au seuil de x
  """
  contingency_table = pd.crosstab(df['target'], df[attr])
  chi2, p_value, degree_o_freedom, expected_freq = scipy.stats.chi2_contingency(contingency_table, correction =False)
  return p_value>=x

class ReducedMLNaiveBayesClassifier(APrioriClassifier) :
  """ utilise le maximum de vraisemblance pour estimer la classe d'un individu en utilisant Naive Bayes """
  def __init__(self, df,x):
    self.keys={str(key):None for key in df.keys()}
    self.P2DL_dic = {attr:P2D_l(df, attr) for attr in self.keys}
    self.df = df
    self.x = x

  def estimProbas(self,attrs):
      probas = {0: None, 1: None}
      target_0 = 1
      target_1 = 1
      #liste des attributs dependants de target 
      attr_dep =dict()

      for attr in attrs.keys():
        if not isIndepFromTarget(self.df, attr, self.x):
          attr_dep[attr] = attrs[attr]

      attrs = attr_dep
      for attr in attrs.keys() :
        if attrs[attr] not in self.P2DL_dic[attr][0].keys(): 
          probas[0]=0
          probas[1]=0
          return probas
        if attr != "target" :
          target_0 *= self.P2DL_dic[attr][0][attrs[attr]]
          target_1 *= self.P2DL_dic[attr][1][attrs[attr]]
      
      probas[0] = target_0
      probas[1] = target_1

      return probas 


  def estimClass(self, attrs):
    probas = self.estimProbas(attrs)
    if probas[0] >= probas[1]:
      return 0
    return 1


  def draw(self):
    args = ""
    attrs = self.df.keys()
    attr_dep = [] #liste des attributs dependants de target

    for attr in attrs :
      #on ne garde que les clés dependantes de Target
      if not isIndepFromTarget(self.df, attr, self.x) :
        attr_dep.append(attr)
      else : 
        print("attributs independants de Target",attr) 
    for fils in attr_dep :
      if(fils != "target"):
        args += attr + "->" +  str(fils) + ";"
    args = args [:-2]
    return utils.drawGraph(args)
  
class ReducedMAPNaiveBayesClassifier(MLNaiveBayesClassifier):
  """
  Un classifier implémente un algorithme pour estimer la classe d'un vecteur d'attributs en tenant en compte que les attributs necessaires pour estimer la classe(ceux dependants). Il propose aussi comme service
  de calculer les statistiques de reconnaissance à partir d'un pandas.dataframe.
  """
  def __init__(self, df,x):
    self.P2DL_attrs = dict()
    for attr in df.keys():
      self.P2DL_attrs[attr] = P2D_l(df,attr)
    self.p = getPrior(df)['estimation']
    self.df = df 
    self.x = x

  def estimProbas(self,attrs):
      probas = {0: None, 1: None}
      target_0 = 1
      target_1 = 1
      #liste des attributs dependants de target 
      attr_dep =dict()

      for attr in attrs.keys():
        if not isIndepFromTarget(self.df, attr, self.x):
          attr_dep[attr] = attrs[attr]
      
      attrs = attr_dep
       #Calcul P(target)
      p_t0 = self.df[(self.df['target'] == 0)].shape[0]/len(self.df) #P(target=0)
      p_t1 = self.df[(self.df['target'] == 1)].shape[0]/len(self.df) #P(target=1)
      for attr in attrs.keys():
        if attrs[attr] not in self.P2DL_attrs[attr][0].keys():
          probas[0] = 0
          probas[1] = 0
          return probas
        if attr != "target":
          target_0 *= self.P2DL_attrs[attr][0][attrs[attr]]
          target_1 *= self.P2DL_attrs[attr][1][attrs[attr]]
      somme = p_t1 *  target_1 + p_t0* target_0
      if somme == 0:
        probas[0] = 0
        probas[1] = 0
        return probas
      probas[0] = target_0*p_t0/somme
      probas[1] = target_1*p_t1/somme
      return probas
  def estimClass(self, attrs):
      probas = self.estimProbas(attrs)
      if probas[0] >= probas[1]:
          return 0
      return 1
  
  def draw(self):
    args = ""
    attrs = self.df.keys()
    attr_dep = [] #liste des attributs dependants de target

    for attr in attrs :
      #on ne garde que les clés dependantes de Target
      if not isIndepFromTarget(self.df, attr, self.x) :
        attr_dep.append(attr)
      else : 
        print("attributs independants de Target",attr) 
    for fils in attr_dep :
      if(fils != "target"):
        args += attr + "->" +  str(fils) + ";"
    args = args [:-2]
    return utils.drawGraph(args)

#===========question 7.2==================
def mapClassifiers(dic,df):
  """
  representation graphique des classifieur en fonction de la precision et du rappel
  @param dic : dictionnaire des classifieurs
  @param  df : donnees
  """
  plt.figure()
  for classifier in dic.keys():
        stats=dic[classifier].statsOnDF(df)
        precision=stats['Précision']
        rappel=stats['Rappel']
        plt.scatter(precision,rappel,marker='x',color='red')
        plt.text(precision,rappel, classifier)
  plt.show()

#===================question 8.1=========================
def MutualInformation(df,X , Y) :
  """calcul de l'information mutuelle I(X, Y) """
  N = len(df)
  XY = np.array(df[[X,Y]])
  X = XY[:,0]
  Y = XY[:,1]
  
  

  val_XY, P_xy = np.unique(XY, axis=0, return_counts=True)
  val_X, P_x = np.unique(X, return_counts=True)
  val_Y, P_y = np.unique(Y, return_counts=True)
  P_xy = P_xy/N
  P_x = P_x/N
  P_y = P_y/N

  
  I = 0
  for [Xi, Yi] in val_XY : 
    i = np.where((val_XY[:,0]==Xi)&(val_XY[:,1]==Yi))[0][0]
    pxy = P_xy[i] 
    i = np.where((val_X==Xi))[0][0]
    px =P_x[i]
    i = np.where((val_Y==Yi))[0][0]
    py =P_y[i]

    if py!=0 and px!=0 and pxy!=0:
      I += pxy * np.log2((pxy)/(px*py))

  return I

def ConditionalMutualInformation(df,X,Y,Z) :
  """calcul de l'information mutuelle I(X, Y, Z) """
  N = len(df)
  XYZ = np.array(df[[X,Y,Z]])
  XZ = XYZ[:,[0,2]]
  YZ = XYZ[:,[1,2]]
  Z = XYZ[:,2]
  
  

  val_XYZ, P_xyz = np.unique(XYZ, axis=0, return_counts=True)
  val_XZ, P_xz = np.unique(XZ,axis=0, return_counts=True)
  val_YZ, P_yz = np.unique(YZ,axis=0, return_counts=True)
  val_Z, P_z = np.unique(Z, return_counts=True)
  P_xyz = P_xyz/N
  P_xz = P_xz/N
  P_yz = P_yz/N
  P_z = P_z/N

  
  I = 0
  for [Xi, Yi, Zi] in val_XYZ : 
    i = np.where((val_XYZ[:,0]==Xi)&(val_XYZ[:,1]==Yi)&(val_XYZ[:,2]==Zi))[0][0]
    pxyz = P_xyz[i] 
    i = np.where((val_XZ[:,0]==Xi)&(val_XZ[:,1]==Zi))[0][0]
    pxz =P_xz[i]
    i = np.where((val_YZ[:,0]==Yi)&(val_YZ[:,1]==Zi))[0][0]
    pyz =P_yz[i]
    i = np.where(val_Z==Zi)[0][0]
    pz =P_z[i]

    if pz!=0 and pyz!=0 and pxz!=0 and pxyz!=0:
      I += pxyz * np.log2((pz*pxyz)/(pxz*pyz))

  return I
#============question 8.2==================
def MeanForSymetricWeights(cmis):
    """
    @param : matrice des infos de dependances
    @return  : moyenne de la matrice
    """
    n = cmis.shape[0]
    return np.sum(cmis) / (n * (n - 1))

def SimplifyConditionalMutualInformationMatrix(cmis):
    """
     @param : matrice des infos de dependances
     @return  : copie de cmis avec les valeurs plus petites que la moyenne ramnées à 0

    """
    moyenne = MeanForSymetricWeights(cmis)
    for i in range(cmis.shape[0]):
        for j in range(cmis.shape[1]):
            if cmis[i, j] < moyenne:
                cmis[i, j] = 0

#===============question 8.3========================
def Kruskal(df, cmis):
    keys = list(df.keys())
    aretes = []
    len = cmis.shape[0]
    for i in range(len):
        for j in range(i, len):
            if cmis[i, j] > 0:
                aretes.append((keys[i], keys[j], cmis[i, j]))
    aretes.sort(reverse=True, key=lambda elem: elem[2])
    resultat = np.zeros(cmis.shape)

    for key_i, key_j, p in aretes:
        i = keys.index(key_i)
        j = keys.index(key_j)
        l = []
        if not are_connected(resultat, i, j, l):
            resultat[i, j] = p
            resultat[j, i] = p

    aretes = []
    len = resultat.shape[0]
    for i in range(len):
        for j in range(i, len):
            if resultat[i, j] > 0:
                aretes.append((keys[i], keys[j], resultat[i, j]))
    aretes.sort(reverse=True, key=lambda elem: elem[2])
    return aretes

def are_connected(A, i, j, l):
    l.append(i)
    if A[i, j] > 0:
        return True
    for k in range( A.shape[0]):
        if (k not in l) and A[i, k] > 0:
            if are_connected(A, k, j, l):
                return True
    return False
  
#============question 8.4===========================
def ConnexSets(liste_arcs) :
  """
  @param : liste des arcs resultat de Kruskal
  @return : foret orienté en fonction des informations mutuelles des attributs
  """
  resultat =[]
  for e , s, poid in liste_arcs :
    flag = True
    for k in range(len(resultat)) :
      if flag and  ((e in resultat[k]) or (s in resultat[k])):
        resultat[k].update({e,s})
        flag = False
    if flag :
      resultat.append({e,s})
  return resultat

def OrientConnexSets(df, liste_arcs, classe) :
  """
  @param df :donnée
  @param  liste_arcs : resultat de kruskal
  @param  classe : classe
  @return  : foret orientée en fonction de la classe 
  """
  vs = set ()
  edges = dict()
  for v1 , v2, weight in liste_arcs :
    vs.add(v1)
    vs.add(v2)
    if v1 not in edges : 
      edges[v1] = set()
    if v2 not in edges :
      edges [v2] = set()
    edges[v1].add(v2)
    edges[v2].add(v1)

    see = {v:False for v in vs }
    foret = []
    #explorer les noeuds de la foret
    def explore (v,connex):
      see[v]= True
      connex.add(v)
      for v1 in edges[v] :
        if not see[v1]:
          foret.append((v,v1))
          explore(v1)

    connexes = ConnexSets(liste_arcs)
    for connex in connexes :
      list_connex = list(connex)
      mi =  np.array([MutualInformation(df,v, classe)for v in list_connex])
      root = list_connex[np.argmax(mi)]
      explore(root)
    return foret
