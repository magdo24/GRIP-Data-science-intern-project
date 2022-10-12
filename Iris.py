#!/usr/bin/env python
# coding: utf-8

# # Name: Mahmoud Hamed Aboelmagd

# ## Importing libraries

# In[ ]:


import pandas as pd 
import numpy as np
import warnings
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
with warnings.catch_warnings():
    warnings.filterwarnings("ignore")


# In[ ]:


iris=pd.read_csv("C:\\Users\\Mahmoud Abo Elmagd\\Downloads\\Iriss.csv")
iris.drop("Id",axis=1,inplace=True) #just eliminate the id from the dimension 


# ## EDA

# In[ ]:


iris.columns


# In[ ]:


iris.describe()


# In[ ]:


iris.head(n=3)


# In[ ]:


iris.groupby("Species").size()


# ## Scatter Matrix

# In[ ]:


from pandas.plotting import scatter_matrix
scatter_matrix(iris.iloc[:,[0,1]])
plt.show()


# In[ ]:


from pandas.plotting import scatter_matrix
scatter_matrix(iris.iloc[:,[2,3]])
plt.show()


# ### We can see that the petals are more clear to be distinced rather than the sepals.

# In[ ]:


x=iris[["PetalLengthCm","PetalWidthCm"]]
x.mean()


# In[ ]:


#preprocessing
from sklearn.preprocessing import StandardScaler
#All the units are in cm (No need for rescaling but we need to transform it so it can be normalized)
scale=StandardScaler()
scale.fit(x)
x_scaled=scale.transform(x) 
type(x_scaled)


# ### Scaled versus original 

# In[ ]:


plt.scatter(x_scaled[:,0],x_scaled[:,1],marker=".",label="Scaled") #ndarray
plt.scatter(x.iloc[:,[0]],x.iloc[:,[1]],marker=".",label="Original")#DataFrame
plt.legend(loc="upper right")
plt.show()


# **scaled should be of zero mean and of one for standard deviation**`

# ## Optimum k clusters

# In[ ]:


from sklearn.cluster import KMeans
#Inertia for different k
inertia=[]
for i in np.arange(1,11):
    km=KMeans(n_clusters=i)
    km.fit(x_scaled)
    inertia.append(km.inertia_)    


# In[ ]:


#plot (Elbow method to find the optimum k)
plt.plot(np.arange(1,11),inertia,marker="o")
plt.xlabel("Num of clusters")
plt.show()


# **As we can see the optimum num of clusters is three which is the same as in the Species feature**

# # The Final plot 

# In[ ]:


k_opt=3
kmeans=KMeans(k_opt)
kmeans.fit(x_scaled)
y_pred=kmeans.predict(x_scaled)
kmeans.inertia_.round(2)


# In[ ]:


#plot scaled data
plt.scatter(x_scaled[:,0],x_scaled[:,1],c=y_pred)
plt.scatter(kmeans.cluster_centers_[:,0],kmeans.cluster_centers_[:,1],marker="*",s=250,edgecolors="k",c=[0,1,2])
plt.xlabel("alcohol")
plt.ylabel("totalphenol")
plt.title("K-means")
plt.show()

