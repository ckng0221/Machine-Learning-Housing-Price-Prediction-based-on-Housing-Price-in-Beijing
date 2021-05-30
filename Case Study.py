#!/usr/bin/env python
# coding: utf-8

# # 3.0 Methodology

# # 3.1 Data Collection

# ## Import Libraries

# In[1]:


import os

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# ## Import dataset

# In[2]:


filename = 'new.csv'
housing_raw = pd.read_csv(filename, encoding= 'unicode_escape')


IMAGES_PATH = 'Images'


# # 3.2 Explorary Data Analysis (EDA)

# ## Examine data type

# In[3]:


print(housing_raw.dtypes)


# ## Examine the basic summary of data

# In[4]:


housing_raw.describe()


# ## View the first 5 rows of the data set

# In[5]:


housing_raw.head()


# ## Examine the destribution of housing price 

# In[6]:


ax = housing_raw.price.plot.hist(bins=1000, 
                                 title='Distribution of Housing Price (Original)');
ax.set_xlabel("Housing Price (RMB/m2)");


# ## Preliminary study of the relationship

# In[7]:


housing_raw.corr(method='pearson')['price'].sort_values(ascending=False)


# It appears that the "communityAverage" has the highest positive linear relationship, whereas the "square" has highest negative linear relationship"

# ## Pairplot 

# Pair plot to visualize the relationship between each feature

# In[8]:


import seaborn as sns;
sns.pairplot(housing_raw[['price', 'communityAverage', 'totalPrice', 'followers',
                      'renovationCondition', 'square', 'Lng', 'Lat']], diag_kind='kde');


# Appears that only 'communityAverage' have somewhat a linear relationship with our target 'Price'

# # 3.1 Data Cleaning 

# ## Remove outliers in the data

# In[9]:


# create a copy of Raw
housing = housing_raw.copy()

# Remove non-real numbers
housing = housing[housing.livingRoom.apply(np.isreal)]

def remove_outlier(df, variable):
    """
    For removing outliers based on IQR method
    
    Parameters
    ----------
    df : pandas dataframe
    variable : str, column name of the dataframe
    """
    # Computing IQR
    Q1 = df[variable].quantile(0.25)
    Q3 = df[variable].quantile(0.75)
    IQR = Q3 - Q1

    # Filtering Values between Q1-1.5IQR and Q3+1.5IQR
    filtered = df.query(f'(@Q1 - 1.5 * @IQR) <= {variable} <= (@Q3 + 1.5 * @IQR)').reset_index(drop=True)
    return filtered

# Remove Outlier based on price
housing = remove_outlier(housing, 'price')

# Drop unneeded columns
housing.drop(['url', 'id', 'Cid'], axis=1, inplace=True)


# ## New distribution of housing price after removal of outliers

# In[10]:


ax = housing.price.plot.hist(bins=1000, 
                        title='Distribution of Housing Price (New)')
ax.set_xlabel("Housing Price (RMB/m2)");


# # 3.4 Data Visualization 

# ## Functions for visualization

# In[11]:


def show_map(df, variable, scale=1., sep='k', tick_n=11, heatmap_label=None):
    """For plotting distribution on map"""
    import matplotlib.image as mpimg
    beijing_img = mpimg.imread('Images/beijingmap_baidu.png') #baidu
    
    ax = df.plot(kind='scatter', x="Lng", y='Lat', alpha=0.04,
                 s=df["followers"]/2, label="Followers",
                 c=variable, cmap=plt.get_cmap("jet"),
                 colorbar=False, figsize=(20, 10))
    plt.imshow(beijing_img, extent=[115.1946, 117.48, 39.495, 40.33], alpha=0.6,   
               cmap=plt.get_cmap("jet")) # baidu

    plt.ylabel("Latitude", fontsize=12)
    plt.xlabel("Longitude", fontsize=12)

    prices = df[variable] * scale
    tick_values = np.linspace(prices.min(), prices.max(), tick_n)
    cbar = plt.colorbar(ticks=tick_values / prices.max(), fraction=0.046, pad=0.04)
    
    def separator(sep):
        if sep == 'k': #thousand separator
            sep_ = 'k'
            div = 1_000
        elif sep == 'M': #million separator
            sep_ ='M'
            div = 1_000_000
        elif sep == None:
            sep_ = ''
            div = 1
            
        cbar.ax.set_yticklabels([f"%d{sep_}"%round((v/div)) for v in tick_values], fontsize=12)
        
    separator(sep)
    cbar.set_label(heatmap_label, fontsize=12)

    plt.legend(fontsize=16)

    plt.xlim([115.9, 116.8])
    plt.ylim([39.5, 40.3])
    plt.show()
    
def save_fig(fig_id, tight_layout=True, fig_extension="png", resolution=300):
    """Save plotted figure"""
    path = os.path.join(IMAGES_PATH, fig_id + "." + fig_extension)
    print("Saving figure", fig_id)
    if tight_layout:
        plt.tight_layout()
    plt.savefig(path, format=fig_extension, dpi=resolution)


# ## Distribution of Housing Price

# In[12]:


# Display the distribution of housing price at Beijing Area
show_map(housing, "price", heatmap_label='House Price [RMB/m2]')


# It appears that the center of the Beijing has the higher price, and the price decreaases as they are moving further away from the center

# ## Visualize for community average and tradeTime

# In[13]:


# groupby the tradeTime & community pivot variables
tradetime_pivot = housing.groupby('tradeTime')['price'].mean()
community_pivot = housing.groupby('communityAverage')['price'].mean()


# In[14]:


import matplotlib.pyplot as plt
import matplotlib.dates as mdates

f1 = plt.figure(figsize=(15, 5))
f2 = plt.figure(figsize=(15, 5))
ax1 = f1.add_subplot(111)
ax1.scatter(tradetime_pivot.index, tradetime_pivot.values,color='red', alpha=0.3)
ax1.set(xlabel='Dates', ylabel='Average Price (RMB)')
ax1.xaxis.set_major_locator(mdates.MinuteLocator(interval=60*24*365)) # # tickmark frequency'
# ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y')) 

ax2 = f2.add_subplot(111)
ax2.scatter(community_pivot.index, community_pivot.values, alpha=0.3)
ax2.set(xlabel='Community Average', ylabel='Average Price (RMB)')
plt.show()


# Shows that the price increase over time, and generally have a linear relationship between average price & community average

# # 3.5 Machine Learning

# # 3.5.1 Feature Engineering

# ## Feature Selection

# In[15]:


house = housing.copy()

# Drop DOM & totalPrice
house.drop(columns=['totalPrice', 'communityAverage'], inplace=True)


# - 'totalPrice' is dropped, as we can directly calculate target unit price based on the 'totalPrice' divided by 'square' the area of the house already, which gives an unfair prediction
# - 'communityAverage' is dropped with similar reason, although no direct relationship, this variable already give a hint what the price range would be, so will make the price predictor less useful 

# ## Feature Transformation

# In[16]:


# Convert Trading time to year only
house.tradeTime = pd.to_datetime(house.tradeTime).dt.strftime("%Y")

# convert desired data to numeric 
house['tradeTime'] = pd.to_numeric(house['tradeTime'])
house['livingRoom'] = house['livingRoom'].apply(pd.to_numeric, errors='coerce')
house['drawingRoom'] = house['drawingRoom'].apply(pd.to_numeric, errors='coerce')
house['bathRoom'] = house['bathRoom'].apply(pd.to_numeric, errors='coerce')
house['constructionTime'] = house['constructionTime'].apply(pd.to_numeric, errors='coerce')

# Convert the floor to numeric
house.floor = house.floor.str.split(' ').str[1].astype('float')


# In[17]:


# Check null data
house.isna().mean().sort_values(ascending=False) 


# ## Check correlation

# In[18]:


## Check correlation
corr = house.corr()['price'].sort_values(ascending=False)
corr


# ## 3.1 Check significance of the correlation

# In[19]:


from scipy.stats.stats import pearsonr

housing_corr = housing.dropna() # dropna for correlation significant test
for column in housing_corr:
    try:
        coefficient, pvalue = pearsonr(housing_corr[column], housing_corr.price)
        if pvalue <= 0.05:
            signif = 'Significant'
        else: 
            signif = 'Not Significant'
            
        print(f"{column} : {pvalue} ({signif})")
    except TypeError:
        pass # pass for non-numeric


# It appears that all the correlation of the variables are significant

# In[20]:


# Prepare the data for features and target for machine learning
X = house.drop('price', axis=1)
y = house['price'].copy()


#  - Drop the 'price' variable, as it is the target that we want to predict
#  - make a copy to the original dataframe, to avoid directly amend on it

# ## Imputation

# In[21]:


# Impute those null values with median
from sklearn.impute import SimpleImputer 
imputer = SimpleImputer(strategy="median")

imputer.fit(X)
X_imp = imputer.transform(X)

X = pd.DataFrame(X_imp, columns=X.columns)


# ## Normalization

# In[22]:


# Normalize the columns value

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()

X = scaler.fit_transform(X)


# ## Test-train split

# In[23]:


# Perform test-train split, to split training set and validation set
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=10)


# # 5.3.2 Development & Testing of Machine Learning Models

# ## Functions for checking accuracy & error

# In[24]:


from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import cross_val_score

def display_scores(model, X_train, y_train, X_test, y_test):
    def rmse(X, y):
        scores = cross_val_score(model, X, y,
                                 scoring="neg_mean_squared_error", cv=10)
        rmse_scores = np.sqrt(-scores)
        return rmse_scores
    rmse_train_scores = rmse(X_train, y_train)
    rmse_test_scores = rmse(X_test, y_test)
    
    # Model Scores
    train_score = model.score(X_train, y_train)
    test_score = model.score(X_test, y_test)
    
    print("Training Set")
    print("-"*50)
    print("Training set: ", train_score)
    print("Scores: ", rmse_train_scores)
    print("Mean: ", rmse_train_scores.mean())
    print("Standard deviation: ", rmse_train_scores.std())
    print("\n"*2)
    print("Testing Set")
    print("-"*50)    
    print("Testing set: ", test_score)
    print("Scores: ", rmse_test_scores)
    print("Mean: ", rmse_test_scores.mean())
    print("Standard deviation: ", rmse_test_scores.std())
    
    y_train_predict = model.predict(X_train)
    y_test_predict = model.predict(X_test)
    
    # Return the predictions & Scores on testing set
    return y_test_predict, rmse_test_scores, test_score


# ## Linear Regression

# In[25]:


get_ipython().run_cell_magic('time', '', 'from sklearn.linear_model import LinearRegression\n\nlin_model = LinearRegression()\nlin_model.fit(X_train, y_train)')


# In[26]:


lin_y_test_predict, lin_test_rmse, lin_test_score = display_scores(lin_model, X_train,
                                                                   y_train, X_test, y_test)


# ## Ridge

# In[27]:


get_ipython().run_cell_magic('time', '', 'from sklearn.linear_model import Ridge\n\nridge_model = Ridge()\nridge_model.fit(X_train, y_train);')


# In[28]:


ridge_y_test_predict, ridge_test_rmse, ridge_test_score = display_scores(ridge_model, X_train, 
                                                                      y_train, X_test, y_test)


# ## Lasso

# In[29]:


get_ipython().run_cell_magic('time', '', 'from sklearn.linear_model import Lasso\n\nlasso_model = Lasso()\nlasso_model.fit(X_train, y_train)')


# In[30]:


lasso_y_test_predict, lasso_test_rmse, lasso_test_score  = display_scores(lasso_model, X_train, 
                                                                          y_train, X_test, y_test)


# Performing relatively bad on both linear and lasso, and ridge model

# ## Decision Tree

# In[101]:


get_ipython().run_cell_magic('time', '', 'from sklearn.tree import DecisionTreeRegressor\n\ntree_model = DecisionTreeRegressor()\ntree_model.fit(X_train, y_train)')


# In[32]:


tree_y_test_predict, tree_test_rmse, tree_test_score = display_scores(tree_model, X_train, 
                                                                      y_train, X_test, y_test)


# ## Random Forest

# In[102]:


get_ipython().run_cell_magic('time', '', "# Random Forest (Full)\n\nfrom sklearn.ensemble import RandomForestRegressor\n\n\nif os.path.isfile('randomforest.sav'):\n    forest_model = load_model('randomforest.sav')\nelse:\n    forest_model = RandomForestRegressor()\n    forest_model.fit(X_train, y_train)")


# In[157]:


forest_y_test_predict, forest_test_rmse, forest_test_score = display_scores(forest_model, X_train, y_train, 
                                                               X_test, y_test)


# In[148]:


#  forest_test_rmse = [7092.28775047, 7244.6936723,  7423.91627079, 7504.32111199, 7507.67920291,
# 7121.99076411, 7337.94371255, 7189.33832306, 7206.38143608, 7275.36728212]

# forest_test_rmse = np.array(forest_test_rmse)


# In[35]:


# Save and load model
import pickle

def save_model(model, filename):
    """Save Machine Learning Model"""
    pickle.dump(model, open(filename, 'wb'))

def load_model(filename):
    """Load Machine Learning Model"""
    model = pickle.load(open(filename, 'rb'))
    return model 

# # for saving the model
# save_model(forest_model, 'randomforest.sav')

# # For loading the model 
# forest_model = load_model('randomforest.sav')


# In[38]:


get_ipython().run_cell_magic('time', '', "## Support Vector Machine\n\nfrom sklearn.svm import SVR\n\nsvm_model = SVR(kernel='rbf')\nsvm_model.fit(X_train, y_train)")


# In[ ]:


svm_y_test_predict, svm_test_rmse, svm_test_score = display_scores(svm_model,X_train, y_train, 
                                                               X_test, y_test)


# In[97]:


get_ipython().run_cell_magic('time', '', '# Gradient Boosting Regressor\nfrom sklearn.ensemble import GradientBoostingRegressor\n\ngb_model = GradientBoostingRegressor(random_state=0)\ngb_model.fit(X_train, y_train)')


# In[98]:


gb_y_test_predict, gb_test_rmse, gb_test_score = display_scores(gb_model, X_train,
                                                                   y_train, X_test, y_test)


# ## Deep Neural Network

# In[34]:


import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers.experimental import preprocessing


# In[35]:


# Perform Normalization for the features
normalizer = preprocessing.Normalization()
normalizer.adapt(X_train)


# In[74]:


def build_and_compile_model(norm):
    model = keras.Sequential([
      norm,
      layers.Dense(15, activation='relu'),
      layers.Dense(15, activation='relu'),
      layers.Dense(10, activation='relu'),
      layers.Dense(5, activation='relu'),
      layers.Dense(1)
    ])

    model.compile(loss='mse',
                  metrics=[tf.keras.metrics.RootMeanSquaredError()],
                  optimizer=tf.keras.optimizers.Adam(0.001))
    return model


# In[75]:


dnn_model = build_and_compile_model(normalizer)
dnn_model.summary()


#  sqrt(input layer nodes * output layer nodes)

# ## Fitting deep neural network model

# In[76]:


get_ipython().run_cell_magic('time', '', '# starts the machine learning\nhistory = dnn_model.fit(\n    X_train, y_train,\n    validation_split=0.2,\n    verbose=1, epochs=300)')


# In[77]:


def plot_loss(history):
    """For ploting the losses over Epoch"""
    plt.figure(figsize=(20, 10))
    plt.plot(history.history['loss'], label='loss')
    plt.plot(history.history['val_loss'], label='val_loss')
    plt.xlabel('Epoch')
    plt.ylabel('Mean Squared Error [price]')
    plt.legend()
    plt.grid(True)


# In[78]:


plot_loss(history)


# ## Collect the results on the test set:

# In[132]:


test_results = {}
test_results['dnn_model'] = dnn_model.evaluate(X_test, y_test, verbose=0)
dnn_test_rmse = test_results['dnn_model'][1]
test_results


# In[80]:


## Prediction
dnn_y_test_predict = dnn_model.predict(X_test).flatten()


# In[93]:


# # Save dnn model
# dnn_model.save('dnn_model_2')

# Load model
# dnn_model = tf.keras.models.load_model('dnn_model')


# In[105]:


import tensorflow_addons as tfa

def coefficient_R_dnn(y_true, y_predict):
    metric = tfa.metrics.r_square.RSquare()
    metric.update_state(y_true, y_predict)
    result = metric.result()
    rsquared = result.numpy()
    return rsquared

dnn_test_score = coefficient_R_dnn(y_test, dnn_y_test_predict)


# In[107]:


dnn_test_score


# # 4.0 Results & Discussion

# ## Overall Results

# In[103]:


forest_test_score = forest_model.score(X_test, y_test)


# In[164]:


# Results
# R squared values 
result_data = {
'RMSE':{
'Linear Regression': lin_test_rmse.mean(),
'Ridge Regression': ridge_test_rmse.mean(),
'Lasso Regression': lasso_test_rmse.mean(),
'Decision Tree': tree_test_rmse.mean(),
'Random Forest': forest_test_rmse.mean(),
'Gradient Boosting': gb_test_rmse.mean(),
'Deep Neural Network': dnn_test_rmse
},
'R Squared':{
'Linear Regression': lin_test_score,
'Ridge Regression': ridge_test_score,
'Lasso Regression': lasso_test_score,
'Decision Tree': tree_test_score,
'Random Forest': forest_test_score,
'Gradient Boosting': gb_test_score,
'Deep Neural Network': dnn_test_score
}
}


# In[166]:


results = pd.DataFrame(result_data).sort_values(by='RMSE')
results


# ## Visualize the predicted results for accuracy

# ### Functions for visualize the prediction Results

# In[95]:


# Function for plot
def match_plot(y_test=y_test, y_test_predict=None, name=None):
    """Plot Prediction & Actual Overlapping plot"""
    num = 100
    z1 = y_test[:num].to_list()
    z2 = y_test_predict[:num]
    
    # for the title for model name
    if name == None:
        name = ""
    else: 
        name = "[" + name + "]"

    df = pd.DataFrame(list(zip(z1, z2)), columns=['Actual', 'Predicted'])
    ax = df.plot(figsize=(10, 5), title=f'{name} Difference Between Actual & Predicted Price')
    ax.set_ylabel("Price (RMB/m2)")
    ax.set_xlabel("Count (Data)")
    return df;

# Function for plot
def plot_actual_predicted(y_test=y_test, y_test_predict=None, name=None):
    """
    Ploting Predicted vs. Actual Price graph
    Parameters
    --------------
    y_test : array, 
    y_predict : predicted y 
    title : title of the chart
    
    """
    if name == None:
        name = ""
    else:
        name = "[" + name + "]"
    
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.scatter(x=y_test, y=y_test_predict, alpha=0.2)
    ax.set(title=f"{name} Actual Price vs. Predicted Price", 
           xlabel='Actual Price (RMB)', ylabel='Predicted Price (RMB)');
    lims = [0, y_test.max()]
    _ = plt.plot(lims, lims, '--k')
    plt.show()
    
# Combine 2 plots
def check_plot(y_test, y_test_predict, name):
    """Display 2 plots together"""
    df = match_plot(y_test, y_test_predict, name)
    plot_actual_predicted(y_test, y_test_predict, name)
    
    return df


# In[70]:


# Linear Regression
lin_df = check_plot(y_test=y_test, y_test_predict=lin_y_test_predict, name='Linear Regression')


# In[71]:


# Lasso
lasso_df = check_plot(y_test, y_test_predict=lasso_y_test_predict, name='Lasso')


# In[74]:


# ridge 
ridge_df = check_plot(y_test, y_test_predict=ridge_y_test_predict, name='Ridge')


# In[75]:


# Decision Tree
tree_df = check_plot(y_test, y_test_predict=tree_y_test_predict, name='Decision Tree')


# In[159]:


# Random Forest
forest_df = check_plot(y_test, y_test_predict=forest_y_test_predict, name='Random Forest')


# In[156]:


# Gradient Boosting
gb_df = check_plot(y_test, y_test_predict=gb_y_test_predict, name='Gradient Boosting')


# In[96]:


# Deep Neural Network
dnn_df = check_plot(y_test, y_test_predict=dnn_y_test_predict, name='Deep Neural Network')


# In[ ]:




