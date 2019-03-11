#!/usr/bin/env python
# coding: utf-8

# #                                            ONLINE RETAIL dataset
#                                                         Mrinmai Sharma
#                                                            11603290
#                                                 Lovely Professional University

# # Libraries used

# In[2]:


import math
import datetime as dt
import numpy as np
import pandas
import pandas as pd
from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import missingno as msno
from sklearn import model_selection
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.preprocessing import scale, StandardScaler, normalize
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn.svm import SVC


# # Load Dataset

# In[3]:


source="OnlineRetail.xlsx"
dataset=pd.read_excel(source)
dataset.head(5)


# # Data Insight

# In[4]:


dataset_copy=dataset
dataset.shape


# In[5]:


dataset.info()


# # Missing data

# In[6]:


msno.matrix(dataset)


# # Cleaning data

# In[7]:


dataset=dataset.loc[pd.isnull(dataset.CustomerID) == False]


# In[8]:


dataset.info()


# In[9]:


msno.matrix(dataset)


# In[11]:


dataset.describe(include='all')


# In[12]:


#remove canceled orders
dataset=dataset[dataset['Quantity']>0]


# In[13]:


#remove rows where customerID are NA
dataset.dropna(subset=['CustomerID'],how='all',inplace=True)


# In[14]:


dataset.shape


# In[ ]:


#Summary


# In[15]:


#exploring the unique values of each attribute
print("Number of transactions: ", dataset['InvoiceNo'].nunique())
print("Number of products bought: ",dataset['StockCode'].nunique())
print("Number of customers:", dataset['CustomerID'].nunique() )
print("Percentage of customers NA: ", round(dataset['CustomerID'].isnull().sum() * 100 / len(dataset),2),"%" )


# # RFM Analysis

# - RECENCY (R): Days since last purchase 
# - FREQUENCY (F): Total number of purchases 
# - MONETARY VALUE (M): Total money this customer spent.

# ## Recency
# To calculate recency, we need to choose a date point from which we evaluate **how many days ago was the customer's last purchase**.

# In[16]:


#last date available in our dataset
dataset['InvoiceDate'].max()


# In[17]:


now = dt.date(2011,12,9)
print(now)


# In[18]:


#create a new column called date which contains the only the date of invoice
dataset['date'] = dataset['InvoiceDate'].dt.date


# In[19]:


dataset.head()


# #### Recency dataset

# In[20]:


#group by customers and check last date of purshace
recency_data = dataset.groupby(by='CustomerID', as_index=False)['date'].max()
recency_data.columns = ['CustomerID','LastPurshaceDate']
recency_data.head()


# In[21]:


#calculate recency
recency_data['Recency'] = recency_data['LastPurshaceDate'].apply(lambda x: (now - x).days)


# In[22]:


recency_data.head()


# In[23]:


#drop LastPurchaseDate as we don't need it anymore
recency_data.drop('LastPurshaceDate',axis=1,inplace=True)


# In[24]:


recency_data.head()


# ## Frequency
# Frequency helps us to know **how many times a customer purchased from us**. To do that we need to check how many invoices are registered by the same customer.

# In[25]:


data_copy=dataset
# drop duplicates
data_copy.drop_duplicates(subset=['InvoiceNo', 'CustomerID'], keep="first", inplace=True)
#calculate frequency of purchases
frequency_data = data_copy.groupby(by=['CustomerID'], as_index=False)['InvoiceNo'].count()
frequency_data.columns = ['CustomerID','Frequency']
frequency_data.head()


# ## Monetary
# 
# Monetary attribute answers the question: **How much money did the customer spent over the time?**

# In[26]:


#create column total cost
dataset['TotalCost'] = dataset['Quantity'] * dataset['UnitPrice']


# In[27]:


monetary_data = dataset.groupby(by='CustomerID',as_index=False).agg({'TotalCost': 'sum'})
monetary_data.columns = ['CustomerID','Monetary']
monetary_data.head()


# ## Create RFM Table
# Merge recency, frequency, monetary data

# In[28]:


rfm_data = recency_data.merge(frequency_data.merge(monetary_data,on='CustomerID'),on='CustomerID')
rfm_data.head()


# In[29]:


#use CustomerID as index
rfm_data.set_index('CustomerID',inplace=True)
rfm_data.head()


# ## Applying 80-20 rule
# Pareto’s rule says **80% of the results come from 20% of the causes**.
# 
# Similarly, **20% customers contribute to 80% of your total revenue**. Let's verify that because that will help us know which customers to focus on when marketing new products.

# In[30]:


#get the 80% of the revenue
pareto_cutoff = rfm_data['Monetary'].sum() * 0.8
print("The 80% of total revenue is: ",round(pareto_cutoff,2))


# In[31]:


customers_rank = rfm_data
# Create a new column that is the rank of the value of coverage in ascending order
customers_rank['Rank'] = customers_rank['Monetary'].rank(ascending=0)
#customers_rank.drop('RevenueRank',axis=1,inplace=True)
customers_rank.head()


# ### Top customers

# In[32]:


customers_rank.sort_values('Rank',ascending=True)


# In[33]:


#get top 20% of the customers
top_20_cutoff = 4339 *0.2
top_20_cutoff


# In[34]:


#sum the monetary values over the customer with rank <=867
revenueByTop20 = customers_rank[customers_rank['Rank'] <= 867]['Monetary'].sum()
revenueByTop20


# The top 20% contribute to more than 80% of the revenue.
# 
# ## Customer segmentation based on RFM score
# I will give a score of 1 to 4 for the data in RFM model.

# 4 -> best/highest value
# 
# 1 -> lowest/worst value

# In[35]:


quantiles = rfm_data.quantile(q=[0.25,0.5,0.75])
quantiles


# In[36]:


quantiles.to_dict()


# **Note:** it is clear that:-
# 
# Higher Recency is bad.
# 
# Higher Frequency, Monetary is profitting.
# and vice-versa.

# In[37]:


# Arguments (x = value, p = recency, monetary_value, frequency, d = quartiles dict)
def RScore(x,p,d):
    if x <= d[p][0.25]:
        return 4
    elif x <= d[p][0.50]:
        return 3
    elif x <= d[p][0.75]: 
        return 2
    else:
        return 1
# Arguments (x = value, p = recency, monetary_value, frequency, k = quartiles dict)
def FMScore(x,p,d):
    if x <= d[p][0.25]:
        return 1
    elif x <= d[p][0.50]:
        return 2
    elif x <= d[p][0.75]: 
        return 3
    else:
        return 4


# In[38]:


#create rfm segmentation table
rfm_segmentation = rfm_data
rfm_segmentation['R_Quartile'] = rfm_segmentation['Recency'].apply(RScore, args=('Recency',quantiles,))
rfm_segmentation['F_Quartile'] = rfm_segmentation['Frequency'].apply(FMScore, args=('Frequency',quantiles,))
rfm_segmentation['M_Quartile'] = rfm_segmentation['Monetary'].apply(FMScore, args=('Monetary',quantiles,))


# In[39]:


rfm_segmentation.head()


# In[40]:


#Combine the RFM scores
rfm_segmentation['RFMScore'] = rfm_segmentation.R_Quartile.map(str) + rfm_segmentation.F_Quartile.map(str) + rfm_segmentation.M_Quartile.map(str)
rfm_segmentation.head()


# Best Recency score = 4: most recently purchase. Best Frequency score = 4: most quantity purchase. Best Monetary score = 4: spent the most.

# In[41]:


#top 10 customers
rfm_segmentation[rfm_segmentation['RFMScore']=='444'].sort_values('Monetary', ascending=False).head(10)


# In[42]:


#Classification based on these scores
print("Best Customers: ",len(rfm_segmentation[rfm_segmentation['RFMScore']=='444']))
print('Loyal Customers: ',len(rfm_segmentation[rfm_segmentation['F_Quartile']==4]))
print("Big Spenders: ",len(rfm_segmentation[rfm_segmentation['M_Quartile']==4]))
print('Almost Lost: ', len(rfm_segmentation[rfm_segmentation['RFMScore']=='244']))
print('Lost Customers: ',len(rfm_segmentation[rfm_segmentation['RFMScore']=='144']))
print('Lost Cheap Customers: ',len(rfm_segmentation[rfm_segmentation['RFMScore']=='111']))


# Now that we knew our customers segments we can choose how to target or deal with each segment.
# 
# For example:
# 
# **Best Customers -**: Reward them. They can be early adopters to new products. Suggest them "Refer a friend".
# 
# **At Risk**: Send them personalized emails to encourage them to shop.
# 
# reference: https://searchsalesforce.techtarget.com/definition/customer-segmentation
# 
# ## Final data for prediction
# Add a column called **CustomerClass** with values ('Best', 'Loyal', 'BigSpender', 'AlmostLost', 'Lost', 'LostCheap')
# 

# In[43]:


def classifier(CustomerID , RFMScore, F_Quartile, M_Quartile, data):
    if(data[RFMScore][CustomerID]=='444'):
        return 'Best'
    elif(data[F_Quartile][CustomerID]==4):
        return 'Loyal'
    elif(data[M_Quartile][CustomerID]==4):
        return 'BigSpenders'
    elif(data[RFMScore][CustomerID]=='244'):
        return 'AlmostLost'
    elif(data[RFMScore][CustomerID]=='144'):
        return 'Lost'
    elif(data[RFMScore][CustomerID]=='111'):
        return 'LostCheap'
    else:
        return 'Others'


# In[44]:


rfm_data.head()


# In[45]:


copy=rfm_data


# In[46]:


copy['CustomerID']=copy.index


# In[47]:


copy['CustomerClass']=copy['CustomerID'].apply(classifier, args=('RFMScore','F_Quartile', 'M_Quartile', rfm_data))


# In[48]:


copy.head(10)


# In[49]:


copy.drop('CustomerID',axis=1,inplace=True)


# In[50]:


copy.head()


# In[51]:


import copy


# In[52]:


final1=copy.deepcopy(rfm_data)


# In[53]:


final2=copy.deepcopy(final1)


# In[54]:


final2.drop('R_Quartile',axis=1,inplace=True)
final2.drop('F_Quartile',axis=1,inplace=True)
final2.drop('M_Quartile',axis=1,inplace=True)
final2.drop('RFMScore',axis=1,inplace=True)


# In[55]:


final1.head(3)


# In[56]:


final2.drop('Rank',axis=1,inplace=True)


# In[57]:


final2.head(3)


# ## Final Datasets:
# final1, final2

# # Using SVM

# In[58]:


final2['CustomerClass'].head()


# In[59]:


final2.drop('CustomerClass',axis=1,inplace=True)


# In[60]:


final2.corr()


# In[61]:


final2['Class']=final1['CustomerClass']
final2.head()


# In[62]:


final2.corr()


# In[63]:


sns.heatmap(final2.corr())


# In[64]:


sns.set_style("whitegrid")
sns.FacetGrid(final2, hue="Class", height=4).map(plt.scatter, "Recency", "Monetary").add_legend()
plt.show()


# In[65]:


scatter_matrix(final2, alpha = 0.3, figsize = (21,10), diagonal = 'kde');


# In[66]:


final2_r_log = np.log(final2['Recency']+0.1)
final2_f_log = np.log(final2['Frequency'])
final2_m_log = np.log(final2['Monetary']+0.1)
final2_c_log = final2['Class']


# In[67]:


log_data=pd.DataFrame({'Monetary': final2_m_log,'Recency': final2_r_log,'Frequency': final2_f_log})


# In[68]:


log_data['Class']=final2['Class']


# In[69]:


log_data.head()


# In[70]:


scatter_matrix(log_data, alpha = 0.3, figsize = (21,10), diagonal = 'kde');


# In[71]:


array=log_data.values


# In[72]:


X = array[:,0:2]
Y = array[:,3]
validation_size = 0.20
seed = 7
X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(X, Y, test_size=validation_size, random_state=seed)


# ## Running SVM for different values of C

# In[75]:


c=0.5;
#while(c<=10):
svm=SVC(kernel='rbf', C = c)
svm.fit(X_train, Y_train)
predictions = svm.predict(X_validation)
correct=0.0
for i in range(len(predictions)):
    if(predictions[i]==Y_validation[i]):
        correct+=1
accuracy=correct/len(predictions)
msg="C= %.1f -> accuracy = %f" % (c,accuracy)
print(msg)
#c+=0.1


# In[74]:


svm=SVC(kernel='linear', C = c)
svm.fit(X_train, Y_train)
predictions = svm.predict(X_validation)
correct=0.0
for i in range(len(predictions)):
    if(predictions[i]==Y_validation[i]):
        correct+=1
accuracy=correct/len(predictions)
msg="C= %.1f -> accuracy = %f" % (c,accuracy)
print(msg)


# In[238]:





# In[ ]:




