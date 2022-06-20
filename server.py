#!/usr/bin/env python
# coding: utf-8

# # **<h1><span style="color:green"><u>Fraude Detection</u></span></h1>**
# **Réalisé par : Hazem-Jonas-Simon**

# In[1]:


#get_ipython().system('pip install imblearn')


# 
# ###  <font color='green'>0. Importation des librairies</font>
# 

# In[2]:


# importer le package pandas
#get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
sns.set_theme()
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans
from imblearn.over_sampling import RandomOverSampler, SMOTE

#on importe le jeu de donnée
df=pd.read_csv(r'/home/ubuntu/exam_scoring/git_exam_churn/fraud.csv',index_col="user_id")


# ## I. Audit des données

# ### 1.Description de la base de donnée

# 

# In[3]:


df.head()


# In[4]:


df['ip_address'] = df['ip_address'].astype('str')
df.head()


# 

# In[5]:


###########################################################################
############### verification du jeux des données ##########################
###########################################################################

df.describe(include='all')


# In[6]:


df.info()


###########################################################################
############### Pas de champs vides dans la tables  #######################
###########################################################################


# ###  <font color='orange'>Observation: Pas de champs vide dans notre table</font>

# In[7]:


print("Le nombre des champs vides par colomne est :")
df.isnull().sum()


# In[8]:


print("Le nombre des champs dupliqué est")
df.duplicated().sum()
#df['device_id'].duplicated().sum()
#df['ip_address'].duplicated().sum()


# ###  <font color='Orange'>Observation: Pas de lignes dupliquées dans notre table</font>

# ### 2. Description des variables

# In[9]:


print("La date de début d'observation est :",min(df["purchase_time"]))
print("La date de fin d'observation est :",max(df["purchase_time"]))


# In[10]:


print("Descrition de la colomne Source")
df["source"].describe()


# In[11]:


df["source"].value_counts()


# In[12]:


df["browser"].value_counts()


# In[13]:


#df["age"].value_counts()
X=np.arange(0, 100, 10)
df1=df['age'].groupby(pd.cut(df["age"],X)).count()
df1


# In[14]:


print("Repartition dans colomne fraude")
df['is_fraud'].value_counts(normalize=1)


# In[15]:


print("Descrition de la colomne fraude")
df["device_id"][df['is_fraud']==1].describe()
#df["device_id"][df['is_fraud']==0].describe()


# In[16]:


print("Descrition de la colomne ip_adress")
df["ip_address"][df['is_fraud']==1].describe()
#df["device_id"][df['is_fraud']==0].describe()


# ### <span style="color:blue"> **3. Analyse statistique descriptive Etape variable numérique**</span> 

# In[17]:


#Analyse statistique descriptive Etape variable numérique

num_data=df.select_dtypes(include=['int64','float64'])


# calcul de la moyenne
stats = pd.DataFrame(num_data.mean(), columns=['moyenne'])
stats.round(2)
# calcul de la mediane
stats['median'] = num_data.median()
stats
# valeur absolue de la différence entre moyenne et médiane
stats['mean_med_diff'] =np.abs(stats['median']-stats['moyenne'])
stats.round(2)
# calcul du quantile
# calcul du quantile
stats['q1'] =num_data.quantile(q=0.25)
stats['q2'] =num_data.quantile(q=0.5)
stats['q3'] =num_data.quantile(q=0.75)
stats
# calcul du min max
stats['min'] =num_data.min()
stats['max'] =num_data.max()
stats['min_max_diff']=stats['max']-stats['min']
stats
# Insérez votre code ici
stats['equart_type'] = num_data.std()
stats


# In[18]:


state_summary = pd.crosstab(df['sex'],df['is_fraud'],normalize=1).round(3)
state_summary


# ###  <font color='Orange'>Observation: Il y a plus de Fraudeurs Masculin que des Fraudeurs Feminin</font>

# In[19]:


X=np.arange(0, 100, 10)
state_summary = pd.crosstab(pd.cut(df["age"],X),df['is_fraud'],normalize=1).round(3)*100
state_summary




# ###  <font color='Orange'>Observation: On a une sur représentation des Fraudeurs dans la tranche d'age 30-40</font>

# In[20]:


state_summary = pd.crosstab(df['source'],df['is_fraud'],normalize=1).round(3)*100
state_summary


# ###  <font color='Orange'>Observation: On a une sur representation des Fraudeurs avec la source Direct</font>

# In[21]:


state_summary = pd.crosstab(df['browser'],df['is_fraud'],normalize=1).round(3)*100
state_summary


# ###  <font color='Orange'>Observation: On a une Sur representation des Fraudeurs avec le Browser Chrome</font>

# ### <span style="color:blue"> **B. Analyse statistique descriptive Etape variable qualitative**</span> 

# In[22]:


#Analyse statistique descriptive Etape variable qualitative

cat_data=df.select_dtypes(exclude= ['int64','float64'])
cat_data = df.select_dtypes(include='O')
cat_data


# In[23]:


###########################################################################
############### Conversion format Date des champs time ####################
###########################################################################
#Conversion format date des champs purchase_time et signup_time
from datetime import datetime
from datetime import date
from datetime import time
from datetime import datetime
from datetime import timedelta
for i in df.index:
    x=datetime.strptime(df['purchase_time'][i],'%Y-%m-%d %H:%M:%S')
    df['purchase_time'][i]=x
for i in df.index:
    x=datetime.strptime(df['signup_time'][i],'%Y-%m-%d %H:%M:%S')
    df['signup_time'][i]=x


# In[24]:


###########################################################################
############### Add new column month, day and Hour #######################
###########################################################################

df['month_purchase']=''
df['day_purchase']=''
df['hour_purchase']=''
for i in df.index:
    x=df['purchase_time'][i].month
    y=df['purchase_time'][i].day
    z=df['purchase_time'][i].hour
    df['month_purchase'][i]=x
    df['day_purchase'][i]=y
    df['hour_purchase'][i]=z

    
df


# In[25]:


###########################################################################
######################## Add new column delais  ###########################
###########################################################################
#Add colomn delay between signup and purchase time
df["delais"]=''
for i in df.index:
    x=df["purchase_time"][i].month*31 + df["purchase_time"][i].day
    y=df["signup_time"][i].month*31+ df["signup_time"][i].day
    df["delais"][i]=x-y


df


# In[26]:


#Analyse des liaisons entre les variables d'un jeu de données 1/2
# Matrice de Correlation de df

print(df.corr())


# ###  <font color='Orange'>Observation: Il y a une faible corrélation entre les variables numérique</font>

# In[27]:


df


# In[28]:


###########################################################################
##### Selection des variables qualitative et calcul des cotrrélations #####
###########################################################################

cat_data = df.select_dtypes(include='O')

cat_data=cat_data[['source','browser']]
#cat_data=cat_data.drop('signup_time',axis=1)
#cat_data=cat_data.drop('purchase_time',axis=1)
table = pd.crosstab(df['source'],df['browser'])
table


# In[29]:



# dépendance des variable qualitative
from scipy.stats import chi2_contingency

resultats_test = chi2_contingency(table)
statistique = resultats_test[0]
p_valeur = resultats_test[1]
degre_liberte = resultats_test[2]

print(statistique)
print(p_valeur)
print(degre_liberte)
print("Test du khi2	< 5%	On rejette H0H0	V de Cramer")


# In[30]:


# fonction V_Cramer qui prend en argument un tableau de contingence, le nombre d'observations et renvoie la valeur du V de Cramer
def V_Cramer(num_data, N):
    stat_chi2 = chi2_contingency(num_data)[0]
    k = table.shape[0]
    r = table.shape[1]
    phi = max(0,(stat_chi2/N)-((k-1)*(r-1)/(N-1)))
    k_corr = k - (np.square(k-1)/(N-1))
    r_corr = r - (np.square(r-1)/(N-1))
    return np.sqrt(phi/min(k_corr - 1,r_corr - 1))

print(V_Cramer(num_data, df.shape[0]))
print("Le V_Cramer n'est pas très élevé,On en déduit qu'il n'y a pas une forte corrélation entre les deux variables mais qu'elle n'est pas non plus négligeable.")


# ###  <font color='Orange'>Observation: Le V_Cramer n'est pas très élevé,On en déduit qu'il n'y a pas une forte corrélation entre les deux variables mais qu'elle n'est pas non plus négligeable.</font>

# ## <span style="color:red"> II. Visualisation des données</span> 

# In[31]:


###########################################################################
#################"## Affichage du nombre des échantillons ################
###########################################################################
sns.countplot(df['is_fraud'])
print('La répartition des cas de fraudes et non fraudes n''est pas équilibré dans notre echantillons de données')


# ###  <font color='Orange'>Observation: La répartition des cas de fraudes et non fraudes n''est pas équilibré dans notre echantillons de données</font>

# In[32]:


###########################################################################
##################### Répartition des Fraudeurs par Age  ###################
###########################################################################

sns.histplot(data=df,x=df['age'][df['is_fraud']==1],bins=12,stat='percent',color='red',binrange=(10,70)).set(title='Distribution des Ages des Fraudeurs')


# In[33]:


sns.histplot(data=df,x=df['age'][df['is_fraud']==0],bins=12,stat='percent',multiple="dodge",color='green',binrange=(10,70)).set(title='Distribution des Ages des non Fraudeurs')


# ###  <font color='Orange'>Observation: Les Fraudeurs sont principalement plus agés que les non fraudeurs</font>

# In[34]:


###########################################################################
################### Répartition des Fraudeurs par Délais  #################
###########################################################################

sns.histplot(data=df,x=df['delais'][df['is_fraud']==1],bins=24,stat='percent',color='red',binrange=(0,120)).set(title='Distribution des delais des Fraudeurs')


# ###  <font color='Orange'>Observation: Les Fraudeurs sont cractérisés par un délais purchase - signup tres petit</font>

# In[35]:


sns.histplot(data=df,x=df['delais'][df['is_fraud']==0],bins=24,stat='percent',color='green',binrange=(0,120)).set(title='Distribution des delais des non Fraudeurs')


# In[36]:


###########################################################################
############### Répartition des Fraudeurs par Purchase Day  ###############
###########################################################################

sns.histplot(data=df,x=df['day_purchase'][df['is_fraud']==1],bins=31,stat='percent',color='red',binrange=(0,31)).set(title='Distribution des jours des Fraudeurs')


# ###  <font color='Orange'>Observation: Les premiers jours du mois sont les jours ou on observe plus de fraudes</font>

# In[37]:


sns.histplot(data=df,x=df['day_purchase'][df['is_fraud']==0],bins=31,stat='percent',color='green',binrange=(0,31)).set(title='Distribution des jours des non Fraudeurs')


# In[38]:


###########################################################################
############# Répartition des Fraudeurs par Purchase Month  ###############
###########################################################################

sns.histplot(data=df,x=df['month_purchase'][df['is_fraud']==1],bins=11,stat='percent',color='red',binrange=(1,12)).set(title='Distribution des mois des Fraudeurs')


# ###  <font color='Orange'>Observation: Le mois de janvier est le mois ou on observe plus de fraudes</font>

# In[39]:


sns.histplot(data=df,x=df['month_purchase'][df['is_fraud']==0],bins=11,stat='percent',color='green',binrange=(1,12)).set(title='Distribution des mois des non Fraudeurs')


# In[40]:


###########################################################################
############### Répartition des Fraudeurs par Purchase Hour  #############
###########################################################################

sns.histplot(data=df,x=df['hour_purchase'][df['is_fraud']==1],bins=24,stat='percent',color='red',binrange=(0,24)).set(title='Distribution des heures des Fraudeurs')


# In[41]:


sns.histplot(data=df,x=df['hour_purchase'][df['is_fraud']==0],bins=24,stat='percent',color='green',binrange=(0,24)).set(title='Distribution des heures des non Fraudeurs')


# In[42]:


###########################################################################
################### Répartition des Fraudeurs par Sex  ####################
###########################################################################

df['sex'][df['is_fraud']==1].value_counts().plot.pie(autopct="%.1f%%");


# In[43]:


df['sex'][df['is_fraud']==0].value_counts().plot.pie(autopct="%.1f%%");


# ###  <font color='Orange'>Observation: Une sur representation du Sex Masculin de 1,3pts pour les Fraudeurs</font>

# In[44]:


df.boxplot(column='purchase_value',by='is_fraud')


# ## Transformation des données

# In[45]:


df.head()
df=df.drop('signup_time',axis=1)
df=df.drop('purchase_time',axis=1)
df


# In[46]:


###########################################################################
#########changement du format en format INT################################
###########################################################################
df['month_purchase'] = df['month_purchase'].astype('int')
df['day_purchase'] = df['day_purchase'].astype('int')
df['hour_purchase'] = df['hour_purchase'].astype('int')
df['delais'] = df['delais'].astype('int')
df.info()


# Transformation des lignes Qualitatives en Columns: Sex, Browser et Source

# In[47]:


##########################################################################
#Transformation des lignes Qualitatives en Columns: Sex, Browser et Source
##########################################################################
df = df.join(pd.get_dummies(df.sex, prefix='sex'))
df=df.drop('sex',axis=1)
df=df.drop('sex_F',axis=1)
df = df.join(pd.get_dummies(df.browser, prefix='browser'))
df=df.drop('browser',axis=1)
df = df.join(pd.get_dummies(df.source, prefix='source'))
df=df.drop('source',axis=1)
df


# In[48]:


df2=df
df2


# In[49]:


###########################################################################
############## Groupement des Classes d'ages ##############################
###########################################################################
bins= [0,30,40,50,70,100]
labels = [1,2,3,4,5]
df2['age'] = pd.cut(df2['age'], bins=bins, labels=labels, right=False)
df2


# In[50]:


###########################################################################
############## Changement d'index, nouveau index device_id ################
###########################################################################
functions_to_apply = {
    # Les méthodes statistiques classiques peuvent être renseignées avec
    # chaines de caractères
    'purchase_value':['mean'],
    'age' : ['max'],
    'ip_address' : ['count'],
    'is_fraud':['sum'],
    'month_purchase':['mean'],
    'day_purchase':['mean'],
    'hour_purchase':['mean'],
    'delais':['mean'],
    'sex_M':['sum'],
    'browser_Chrome':['sum'],
    'browser_FireFox':['sum'],
    'browser_IE':['sum'],
    'browser_Opera':['sum'],
    'browser_Safari':['sum'],
    'source_Ads':['sum'],
    'source_Direct':['sum'],
    'source_SEO':['mean'],
    'browser_Safari':['mean']
}
df2=df.groupby('device_id').agg(functions_to_apply)
df2.columns = df2.columns.droplevel(1)
df2.info()


# In[51]:


###########################################################################
############## Normalisation des champs avec données Binaire ##############
###########################################################################

df2['is_fraud'][df2['is_fraud'] > 0] = 1
df2['sex_M'][df2['sex_M'] > 0] = 1
df2['browser_Chrome'][df2['browser_Chrome'] > 0] = 1
df2['browser_FireFox'][df2['browser_FireFox'] > 0] = 1
df2['browser_IE'][df2['browser_IE'] > 0] = 1
df2['browser_Opera'][df2['browser_Opera'] > 0] = 1
df2['browser_Safari'][df2['browser_Safari'] > 0] = 1
df2['source_Ads'][df2['source_Ads'] > 0] = 1
df2['source_Direct'][df2['source_Direct'] > 0] = 1
df2['source_SEO'][df2['source_SEO'] > 0] = 1
df2


# In[52]:


###########################################################################
############## Groupement des Classes prchase value #######################
###########################################################################
bins= [0,20,40,60,80,100,1000]
labels = [1,2,3,4,5,6]
df2['purchase_value'] = pd.cut(df2['purchase_value'], bins=bins, labels=labels, right=False)
df2


# In[52]:





# ### Correction du Problème Imbalanced Classification

# In[53]:


data=df2.drop('is_fraud',axis=1)
target=df2['is_fraud']


# In[54]:


###########################################################################
############### jeu d'entraînement et de test #############################
###########################################################################

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(data, target, test_size = 0.3,random_state = 1)


# In[55]:


###########################################################################
######## Problème Imbalanced Classification: RandomOverSampl ##############
###########################################################################

rOs = RandomOverSampler()
X_ro, y_ro = rOs.fit_resample(X_train, y_train)
print('Classes échantillon oversampled :', dict(pd.Series(y_ro).value_counts()))


# ### Modèle de Régression Logistique

# In[56]:


###########################################################################
############### On importe la classe logisticregression ###################
###########################################################################

from sklearn.linear_model import LogisticRegression 

###########################################################################
############### Instanciation du modèle ###################################
###########################################################################


logreg = LogisticRegression()       

###########################################################################
######### Entraînement du modèle sur le jeu d'entraînement ################
###########################################################################

#logreg.fit(X_train, y_train)   
logreg.fit(X_ro, y_ro)      

###########################################################################
######### Prédiction de la variable cible pour le jeu de données test. ####
######### Ces prédictions sont stockées dans y_pred #######################
###########################################################################
# Prédiction de la variable cible pour le jeu de données test. Ces prédictions sont stockées dans y_pred
y_pred_train = logreg.predict(X_train)
y_pred_test = logreg.predict(X_test)


# In[57]:


# On affiche les coefficients obtenus
coeff=logreg.coef_
# On affiche la constante
intercept=logreg.intercept_
print("la valeur de l'intercept est : ",intercept)
# On calcule les odd-ratios
# On importe la librairie numpy
import numpy as np 
# On calcule les odd ratios
odd_ratios=np.exp(logreg.coef_)
resultats=pd.DataFrame(data.columns, columns=["Variables"])
resultats['Coefficients']=logreg.coef_.tolist()[0]
resultats['Odd_Ratios']=np.exp(logreg.coef_).tolist()[0]
resultats


# ### Évaluation du modèle de classification

# In[58]:


###########################################################################
########### Prédiction de la variable cible pour le jeu de données  #######
###########################################################################

y_pred_train   = logreg.predict(X_train)
y_pred_test   = logreg.predict(X_test)


###########################################################################
#################### Calcul de Matrice de Confusion  ######################
###########################################################################


from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test,y_pred_test)
M=pd.DataFrame(cm)
M


# In[59]:


###########################################################################
############### Calcul des Scores pour evaluation du modéle  ##############
###########################################################################


from sklearn.metrics import accuracy_score
from sklearn import metrics

print("Accuracy:",accuracy_score(y_test,y_pred_test))
print("precision_score:",metrics.precision_score(y_test, y_pred_test))
print("recall_score:",metrics.recall_score(y_test, y_pred_test))
print("f1_score:",metrics.f1_score(y_test, y_pred_test))


# ## 2eme Model : KNN Model

# In[60]:


# methode du coude
# On définit une liste de k 

K = np.arange(1,15).tolist()
liste=[]

# A l'aide d'une boucle, on ajoute à une liste la somme des distances pour chaque k

for k in K:
    clust = KMeans(n_clusters=k, random_state=1).fit(X_train)
    liste.append(clust.inertia_)
    
import matplotlib.pyplot as plt

#get_ipython().run_line_magic('matplotlib', 'inline')

fig,ax=plt.subplots(dpi=130)

# On retire l'axe supérieur et l'axe de droite du graphique

ax.spines['right'].set_visible(False)

ax.spines['top'].set_visible(False)

# On conserve les graduations de l'axe du bas et de gauche

ax.yaxis.set_ticks_position('left') 

ax.xaxis.set_ticks_position('bottom') 

# On affiche le graphique

plt.plot(K, liste)

# On définit le nom des axes

plt.xlabel("Nombre de clusters (k)")
plt.ylabel("Distance")
plt.title('\nDétermination du nombre de clusters optimal par la méthode du coude\n')
plt.show();


# ###  <font color='Orange'>Observation: d'aprés la methode, nous avons 2 ou 3 clusters</font>

# In[61]:


###########################################################################
#############################  Modéle KNN avec K=2 ########################
###########################################################################

## d'aprés la methode, nous avons 2 ou 3 clusters
knn = KNeighborsClassifier(2)

#knn.fit(X_train, y_train)
knn.fit(X_ro, y_ro)

y_pred_knn = knn.predict(X_test)


# In[62]:


###########################################################################
#################### Calcul de Matrice de Confusion  ######################
###########################################################################
from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test,y_pred_knn)
M=pd.DataFrame(cm)
M


# In[63]:


###########################################################################
############### Calcul des Scores pour evaluation du modéle  ##############
###########################################################################

from sklearn.metrics import accuracy_score
from sklearn import metrics
# On affiche l'accuracy du modèle 
print("Accuracy:",accuracy_score(y_test,y_pred_knn))
print("precision_score:",metrics.precision_score(y_test, y_pred_knn))
print("recall_score:",metrics.recall_score(y_test, y_pred_knn))
print("f1_score:",metrics.f1_score(y_test, y_pred_knn))


# In[64]:


###########################################################################
#############################  Modéle KNN avec K=3 ########################
###########################################################################
knn = KNeighborsClassifier(3)

#knn.fit(X_train, y_train)
knn.fit(X_ro, y_ro)

y_pred_knn = knn.predict(X_test)


# In[65]:


###########################################################################
#################### Calcul de Matrice de Confusion  ######################
###########################################################################
from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test,y_pred_knn)
M=pd.DataFrame(cm)
M


# In[66]:


###########################################################################
############### Calcul des Scores pour evaluation du modéle  ##############
###########################################################################
from sklearn.metrics import accuracy_score
from sklearn import metrics
# On affiche l'accuracy du modèle 
print("Accuracy:",accuracy_score(y_test,y_pred_knn))
print("precision_score:",metrics.precision_score(y_test, y_pred_knn))
print("recall_score:",metrics.recall_score(y_test, y_pred_knn))
print("f1_score:",metrics.f1_score(y_test, y_pred_knn))


# In[67]:


# On affiche l'accuracy du modèle 
print("Accuracy:",accuracy_score(y_test,y_pred_knn))
print("precision_score:",metrics.precision_score(y_test, y_pred_knn))
print("recall_score:",metrics.recall_score(y_test, y_pred_knn))
print("f1_score:",metrics.f1_score(y_test, y_pred_knn))


# ## Modéle 3 : Support Vector Machines (SVM Model)

# In[68]:


#Import svm model
from sklearn import svm

#Create a svm Classifier
clf = svm.SVC()
#clf = svm.SVC(kernel='linear') # Linear Kernel


# In[69]:


X_train_svm, X_test_svm, y_train_svm, y_test_svm = train_test_split(data, target, test_size=0.3,random_state=109) # 70% training and 30% test


# In[70]:


#Train the model
#clf.fit(X_train_svm, y_train_svm)
#clf.fit(X_train, y_trainm)
clf.fit(X_ro, y_ro)


# In[71]:


#Predict the response 
#y_pred_svm = clf.predict(X_test_svm)
y_pred_svm = clf.predict(X_test)


# In[72]:


###########################################################################
#################### Calcul de Matrice de Confusion  ######################
###########################################################################
from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test,y_pred_svm)
M=pd.DataFrame(cm)
M


# In[73]:


###########################################################################
############### Calcul des Scores pour evaluation du modéle  ##############
###########################################################################
print("Accuracy:",accuracy_score(y_test_svm,y_pred_svm))
print("precision_score:",metrics.precision_score(y_test_svm, y_pred_svm))
print("recall_score:",metrics.recall_score(y_test_svm, y_pred_svm))
print("f1_score:",metrics.f1_score(y_test_svm, y_pred_svm))


# ###  <font color='Orange'>Observation: Avec le modéle Logistic Regression nous avons observé les méilleurs performancer en comparant les KPI de scoring</font> 

# <a style='text-decoration:none;line-height:16px;display:flex;color:#5B5B62;padding:10px;justify-content:end;' href='https://deepnote.com?utm_source=created-in-deepnote-cell&projectId=cf24841e-9d93-46ae-be33-9a2f9e75abb7' target="_blank">
# <img alt='Created in deepnote.com' style='display:inline;max-height:16px;margin:0px;margin-right:7.5px;' src='data:image/svg+xml;base64,PD94bWwgdmVyc2lvbj0iMS4wIiBlbmNvZGluZz0iVVRGLTgiPz4KPHN2ZyB3aWR0aD0iODBweCIgaGVpZ2h0PSI4MHB4IiB2aWV3Qm94PSIwIDAgODAgODAiIHZlcnNpb249IjEuMSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIiB4bWxuczp4bGluaz0iaHR0cDovL3d3dy53My5vcmcvMTk5OS94bGluayI+CiAgICA8IS0tIEdlbmVyYXRvcjogU2tldGNoIDU0LjEgKDc2NDkwKSAtIGh0dHBzOi8vc2tldGNoYXBwLmNvbSAtLT4KICAgIDx0aXRsZT5Hcm91cCAzPC90aXRsZT4KICAgIDxkZXNjPkNyZWF0ZWQgd2l0aCBTa2V0Y2guPC9kZXNjPgogICAgPGcgaWQ9IkxhbmRpbmciIHN0cm9rZT0ibm9uZSIgc3Ryb2tlLXdpZHRoPSIxIiBmaWxsPSJub25lIiBmaWxsLXJ1bGU9ImV2ZW5vZGQiPgogICAgICAgIDxnIGlkPSJBcnRib2FyZCIgdHJhbnNmb3JtPSJ0cmFuc2xhdGUoLTEyMzUuMDAwMDAwLCAtNzkuMDAwMDAwKSI+CiAgICAgICAgICAgIDxnIGlkPSJHcm91cC0zIiB0cmFuc2Zvcm09InRyYW5zbGF0ZSgxMjM1LjAwMDAwMCwgNzkuMDAwMDAwKSI+CiAgICAgICAgICAgICAgICA8cG9seWdvbiBpZD0iUGF0aC0yMCIgZmlsbD0iIzAyNjVCNCIgcG9pbnRzPSIyLjM3NjIzNzYyIDgwIDM4LjA0NzY2NjcgODAgNTcuODIxNzgyMiA3My44MDU3NTkyIDU3LjgyMTc4MjIgMzIuNzU5MjczOSAzOS4xNDAyMjc4IDMxLjY4MzE2ODMiPjwvcG9seWdvbj4KICAgICAgICAgICAgICAgIDxwYXRoIGQ9Ik0zNS4wMDc3MTgsODAgQzQyLjkwNjIwMDcsNzYuNDU0OTM1OCA0Ny41NjQ5MTY3LDcxLjU0MjI2NzEgNDguOTgzODY2LDY1LjI2MTk5MzkgQzUxLjExMjI4OTksNTUuODQxNTg0MiA0MS42NzcxNzk1LDQ5LjIxMjIyODQgMjUuNjIzOTg0Niw0OS4yMTIyMjg0IEMyNS40ODQ5Mjg5LDQ5LjEyNjg0NDggMjkuODI2MTI5Niw0My4yODM4MjQ4IDM4LjY0NzU4NjksMzEuNjgzMTY4MyBMNzIuODcxMjg3MSwzMi41NTQ0MjUgTDY1LjI4MDk3Myw2Ny42NzYzNDIxIEw1MS4xMTIyODk5LDc3LjM3NjE0NCBMMzUuMDA3NzE4LDgwIFoiIGlkPSJQYXRoLTIyIiBmaWxsPSIjMDAyODY4Ij48L3BhdGg+CiAgICAgICAgICAgICAgICA8cGF0aCBkPSJNMCwzNy43MzA0NDA1IEwyNy4xMTQ1MzcsMC4yNTcxMTE0MzYgQzYyLjM3MTUxMjMsLTEuOTkwNzE3MDEgODAsMTAuNTAwMzkyNyA4MCwzNy43MzA0NDA1IEM4MCw2NC45NjA0ODgyIDY0Ljc3NjUwMzgsNzkuMDUwMzQxNCAzNC4zMjk1MTEzLDgwIEM0Ny4wNTUzNDg5LDc3LjU2NzA4MDggNTMuNDE4MjY3Nyw3MC4zMTM2MTAzIDUzLjQxODI2NzcsNTguMjM5NTg4NSBDNTMuNDE4MjY3Nyw0MC4xMjg1NTU3IDM2LjMwMzk1NDQsMzcuNzMwNDQwNSAyNS4yMjc0MTcsMzcuNzMwNDQwNSBDMTcuODQzMDU4NiwzNy43MzA0NDA1IDkuNDMzOTE5NjYsMzcuNzMwNDQwNSAwLDM3LjczMDQ0MDUgWiIgaWQ9IlBhdGgtMTkiIGZpbGw9IiMzNzkzRUYiPjwvcGF0aD4KICAgICAgICAgICAgPC9nPgogICAgICAgIDwvZz4KICAgIDwvZz4KPC9zdmc+' > </img>
# Created in <span style='font-weight:600;margin-left:4px;'>Deepnote</span></a>
