#!/usr/bin/env python
# coding: utf-8

# #Laptop Price Prediction

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from statsmodels.stats.outliers_influence \
    import variance_inflation_factor
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics \
    import r2_score, mean_absolute_error, mean_squared_error
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression
import pickle


path = "Dataset.csv"
df = pd.read_csv(path, encoding="latin-1")


# ##Data Health Review

# In[66]:


df.sample(5)


# In[67]:


df.info()


# In[68]:


# Checking percentage of missing values
(df.isnull().mean() * 100).sort_values(ascending=False)


# In[69]:


df.duplicated().sum()


# In[70]:


df.describe()


# In[71]:


df.drop(columns=["laptop_ID", "Product"], inplace=True)


# ## Data Analysis & Feature Engineering

# ####Analysis by RAM

# In[72]:


# Extract the numeric values from the Ram column using regular expressions
df["Ram"] = df["Ram"].str.extract(r"(\d+)").astype(int)

sns.barplot(x="Ram", y="Price_euros", data=df)
plt.show()


# ####Analysis by Weight

# In[73]:


# Extract the numeric values from the Weight column using regular expressions
pattern = r"\d+\.?\d*|\.\d+"

df["Weight"] = df["Weight"].str.extract(r"(\d+(?:\.\d+)?)").astype(float)

df["Weight"] = np.log(df["Weight"])

sns.pairplot(
    df, x_vars=["Weight"], y_vars=["Price_euros"], kind="scatter",
    height=5, aspect=1.5
)
plt.show()


# ####Analysis by Screen Resolution

# In[74]:


df["ScreenResolution"].value_counts()


# In[75]:


# how many laptops in data are touchscreen
df["Touchscreen"] = df["ScreenResolution"].apply(
    lambda x: 1 if "Touchscreen" in x else 0
)
sns.barplot(x=df["Touchscreen"], y=df["Price_euros"])


# In[76]:


# extract IPS column
df["Ips"] = df["ScreenResolution"].apply(lambda x: 1 if "IPS" in x else 0)
sns.barplot(x=df["Ips"], y=df["Price_euros"])


# In[77]:


def findXresolution(s):
    return s.split()[-1].split("x")[0]


def findYresolution(s):
    return s.split()[-1].split("x")[1]


# finding the x_res and y_res from screen resolution
df["X_res"] = df["ScreenResolution"].apply(lambda x: findXresolution(x))
df["Y_res"] = df["ScreenResolution"].apply(lambda y: findYresolution(y))

# convert to numeric
df["X_res"] = df["X_res"].astype("int")
df["Y_res"] = df["Y_res"].astype("int")

# Replacing inches, X and Y resolution to PPI
df["ppi"] = (((df["X_res"] ** 2) +
              (df["Y_res"] ** 2)) ** 0.5 / df["Inches"]).astype(
    "float"
)
df.corr()["Price_euros"].sort_values(ascending=False)

df.drop(columns=["ScreenResolution", "Inches", "X_res", "Y_res"],
        inplace=True)


# In[78]:


df["ppi"] = np.log(df["ppi"])
sns.pairplot(
    df, x_vars=["ppi"], y_vars=["Price_euros"], kind="scatter",
    height=5, aspect=1.5
)


# ####Analysis by Processor brand

# In[79]:


def fetch_processor(x):
    cpu_name = " ".join(x.split()[0:3])
    if (
        cpu_name == "Intel Core i7"
        or cpu_name == "Intel Core i5"
        or cpu_name == "Intel Core i3"
    ):
        return cpu_name
    elif cpu_name.split()[0] == "Intel":
        return "Other Intel Processor"
    elif cpu_name.split()[0] == "AMD":
        return "AMD Processor"
    else:
        return "Samsung processor"


df["Cpu_brand"] = df["Cpu"].apply(lambda x: fetch_processor(x))

sns.barplot(x=df["Cpu_brand"], y=df["Price_euros"])
plt.xticks(rotation="vertical")
plt.show()
df.drop(columns=["Cpu"], inplace=True)


# ####Analysis by Memory

# In[80]:


df["Memory"] = df["Memory"].astype(str).replace(r"\.0", "", regex=True)
df["Memory"] = df["Memory"].str.replace("GB", "")
df["Memory"] = df["Memory"].str.replace("TB", "000")
new = df["Memory"].str.split("+", n=1, expand=True)

df["first"] = new[0]
df["first"] = df["first"].str.strip()

df["second"] = new[1]

df["Layer1HDD"] = df["first"].apply(lambda x: 1 if "HDD" in x else 0)
df["Layer1SSD"] = df["first"].apply(lambda x: 1 if "SSD" in x else 0)
df["Layer1Hybrid"] = df["first"].apply(lambda x: 1 if "Hybrid" in x else 0)
df["Layer1Flash_Storage"] = df["first"].apply(
    lambda x: 1 if "Flash Storage" in x else 0
)

df["first"] = df["first"].str.replace(r"\D", "", regex=True)

df["second"].fillna("0", inplace=True)

df["Layer2HDD"] = df["second"].apply(lambda x: 1 if "HDD" in x else 0)
df["Layer2SSD"] = df["second"].apply(lambda x: 1 if "SSD" in x else 0)
df["Layer2Hybrid"] = df["second"].apply(lambda x: 1 if "Hybrid" in x else 0)
df["Layer2Flash_Storage"] = df["second"].apply(
    lambda x: 1 if "Flash Storage" in x else 0
)

df["second"] = df["second"].str.replace(r"\D", "", regex=True)

df["first"] = df["first"].astype(int)
df["second"] = df["second"].astype(int)

df["HDD"] = (df["first"] * df["Layer1HDD"] +
             df["second"] * df["Layer2HDD"])
df["SSD"] = (df["first"] * df["Layer1SSD"] +
             df["second"] * df["Layer2SSD"])
df["Hybrid"] = (df["first"] * df["Layer1Hybrid"] +
                df["second"] * df["Layer2Hybrid"])
df["Flash_Storage"] = (df["first"] * df["Layer1Flash_Storage"] +
                       df["second"] * df["Layer2Flash_Storage"])


df.drop(
    columns=[
        "first",
        "second",
        "Layer1HDD",
        "Layer1SSD",
        "Layer1Hybrid",
        "Layer1Flash_Storage",
        "Layer2HDD",
        "Layer2SSD",
        "Layer2Hybrid",
        "Layer2Flash_Storage",
    ],
    inplace=True,
)


# In[81]:


df.drop(columns=["Memory", "Hybrid", "Flash_Storage"], inplace=True)


# In[82]:


df.sample(5)


# ####Analysis by Gpu

# In[83]:


df["Gpu"].value_counts()


# In[84]:


# Which brand GPU is in laptop
df["Gpu_brand"] = df["Gpu"].apply(lambda x: x.split()[0])
# there is only 1 row of ARM GPU so remove it
df = df[df["Gpu_brand"] != "ARM"]
df.drop(columns=["Gpu"], inplace=True)

sns.barplot(x=df["Gpu_brand"], y=df["Price_euros"])
plt.xticks(rotation="vertical")
plt.show()


# #### Analysis by Company

# In[85]:


counts = df["Company"].value_counts()
mask = counts < 5
to_remove = counts[mask].index.tolist()
df = df[df["Company"].isin(to_remove)]
sns.catplot(
    x="Company", y="Price_euros", kind="bar", data=df, height=5, aspect=1.5
).set_xticklabels(rotation=90)


# ####Analysis by Typename & OpSys

# In[41]:


sns.catplot(x="TypeName", y="Price_euros",
            kind="bar", data=df).set_xticklabels(rotation=90)
sns.catplot(x="OpSys", y="Price_euros", kind="bar", data=df).set_xticklabels(
    rotation=90
)

plt.show()


#

# #### Summary of Data Analysis
#
# As our primary motive is to train, test and predict the price
# of the laptops based on this dataset,a comprehensive Analysis
# of the various variables of the dataset are done against the Price data.
#
# These are the changes made in order to make the analysis more efficient
# and effective:
#
#
#
# 1. The Data in **Ram & Weight** variables were made numerical by
# removing their respective units
# 2. The **ScreenResolution** Varaible has been split into 3 variables-
# **Tourchscreen, Ips & ppi(resolution) **
# 3. Cpu & Gpu variables were used to get Cpu_brand & Gpu_brand variables
# 4. The **Memory** Variable were used to get various information and
# finally made into two variables- **SSD & HDD**, containing the size
# of the memory in each respectively.
# 5. The variable Company had **more unique values** and few of those were
# made very few observations in the dataset, as removing such least
# observed company data help us to reduce the overfitting while performing
# the Linear regression through encoding, we have removed company data
# that had less than 5 observations
# 6. Later, the variables - **Memory, Cpu, Gpu, ScreenResolution** were
# removed, while the variables **Laptop ID & Product** were removed as
# they had least use for the model building.
# 7. The distribution of data of variable Weight is normalised using log
# function to have a better distribution of the data.
#

# In[86]:


categorical = [col for col in df.columns if df[col].dtypes == "O"]
print("The Categorical variables are : \n ", categorical)


# In[87]:


Numerical = [
    col
    for col in df.columns
    if (df[col].dtypes == "int64") or (df[col].dtypes == "float64")
]
print("The Numerical variables are : \n ", Numerical)


# In[88]:


df.sample(3)


# ##Model Building

# ####Correlation & VIF

# In[89]:

corr = df.corr()
print(corr)

plt.figure(figsize=(16, 6))
colormap = sns.color_palette("Blues")
sns.heatmap(df.corr(), annot=True, cmap=colormap).set_title(
    "Correlation Heatmap", fontdict={"fontsize": 14}
)
plt.show()


# In[90]:


X = df[
    [
        col
        for col in df.columns
        if (df[col].dtypes == "int64") or (df[col].dtypes == "float64")
    ]
]
# calculate the VIF for each variable
vif = pd.DataFrame()
vif["features"] = X.columns
vif["VIF"] = [variance_inflation_factor(X.values, i)
              for i in range(X.shape[1])]

# display the VIF values
print(vif)

X = df[["Ram", "Weight", "Price_euros", "Touchscreen", "Ips",
        "ppi", "SSD", "HDD"]]
y = df["Price_euros"]
X = sm.add_constant(X)
model = sm.OLS(y, X).fit()

print(model.summary())


X = X.drop(columns=["Ram", "Weight", "Price_euros", "const"])
# calculate the VIF for each variable
vif = pd.DataFrame()
vif["features"] = X.columns
vif["VIF"] = [variance_inflation_factor(X.values, i)
              for i in range(X.shape[1])]
# display the VIF values
print(vif)

X = df[["Touchscreen", "Ips", "ppi", "SSD", "HDD"]]
y = df["Price_euros"]
model = sm.OLS(y, X).fit()
print(model.summary())


# ##Models

df.head()


# ###Simple Linear Regression

# ####Model 1: Vs Company

# In[96]:

# define the predictor variable and response variable
X = df[["Company"]]
y = df["Price_euros"]

# one-hot encode the categorical variable
ct = ColumnTransformer(
    transformers=[("encoder", OneHotEncoder(), [0])],
    remainder="pasthrough"
)
X = ct.fit_transform(X)

# split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.15, random_state=3
)

# train the linear regression model
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# make predictions on the test data
y_pred = regressor.predict(X_test)

accuracies = cross_val_score(estimator=regressor,
                             X=X_train, y=y_train, cv=5)

print(f"R2 Score : {r2_score(y_test, y_pred)*100:.2f}%")
print(f"MAE : {mean_absolute_error(y_test, y_pred)*100:.2f}%")
print(f"MSE : {mean_squared_error(y_test, y_pred)*100:.2f}%")
print("Cross Val Accuracy: {:.2f} %".format(accuracies.mean()
                                            * 100))

model_comparison = {}
model_comparison["SLR Vs Company"] = [
    r2_score(y_test, y_pred),
    mean_squared_error(y_test, y_pred),
    mean_absolute_error(y_test, y_pred),
    (accuracies.mean()),
]


# ####Model 2: Vs Ram

# In[97]:


# Define the predictor variable and response variable
X = df["Ram"]
y = df["Price_euros"]

# Add a constant term to the predictor variable to fit
# an intercept
X = sm.add_constant(X)

# Fit the linear regression model
model = sm.OLS(y, X).fit()

# Print the model summary
print(model.summary())

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.15, random_state=3
)

model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

accuracies = cross_val_score(estimator=model, X=X_train,
                             y=y_train, cv=5)

print(f"R2 Score : {r2_score(y_test, y_pred)*100:.2f}%")
print(f"MAE : {mean_absolute_error(y_test, y_pred)*100:.2f}%")
print(f"MSE : {mean_squared_error(y_test, y_pred)*100:.2f}%")
print("Cross Val Accuracy: {:.2f} %".format(accuracies.mean()
                                            * 100))

model_comparison["SLR Vs Ram"] = [
    r2_score(y_test, y_pred),
    mean_squared_error(y_test, y_pred),
    mean_absolute_error(y_test, y_pred),
    (accuracies.mean()),
]


# ####Model 3: Vs ppi

# In[98]:


# Define the predictor variable and response variable
X = df["ppi"]
y = df["Price_euros"]

# Add a constant term to the predictor variable to fit an intercept
X = sm.add_constant(X)

# Fit the linear regression model
model = sm.OLS(y, X).fit()

# Print the model summary
print(model.summary())

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.15, random_state=3
)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

accuracies = cross_val_score(estimator=model,
                             X=X_train, y=y_train, cv=5)

print(f"R2 Score : {r2_score(y_test,y_pred)*100:.2f}%")
print(f"MAE : {mean_absolute_error(y_test,y_pred)*100:.2f}%")
print(f"MSE : {mean_squared_error(y_test,y_pred)*100:.2f}%")
print("Cross Val Accuracy: {:.2f} %".format(accuracies.mean()
                                            * 100))

model_comparison["SLR Vs ppi"] = [
    r2_score(y_test, y_pred),
    mean_squared_error(y_test, y_pred),
    mean_absolute_error(y_test, y_pred),
    (accuracies.mean()),
]


# ####Model 4: Vs Weight

# In[99]:


# Define the predictor variable and response variable
X = df["Weight"]
y = df["Price_euros"]

# Add a constant term to the predictor variable to
# fit an intercept
X = sm.add_constant(X)

# Fit the linear regression model
model = sm.OLS(y, X).fit()

# Print the model summary
print(model.summary())

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.15, random_state=3
)

model = LinearRegression()
model.fit(X_train, y_train)


y_pred = model.predict(X_test)

accuracies = cross_val_score(estimator=model,
                             X=X_train, y=y_train, cv=5)

print(f"R2 Score : {r2_score(y_test,y_pred)*100:.2f}%")
print(f"MAE : {mean_absolute_error(y_test,y_pred)*100:.2f}%")
print(f"MSE : {mean_squared_error(y_test,y_pred)*100:.2f}%")
print("Cross Val Accuracy: {:.2f} %".format(accuracies.mean()
                                            * 100))

model_comparison["SLR Vs Weight"] = [
    r2_score(y_test, y_pred),
    mean_squared_error(y_test, y_pred),
    mean_absolute_error(y_test, y_pred),
    (accuracies.mean()),
]


# ####Model 5: Vs HDD

# In[100]:


# Define the predictor variable and response variable
X = df["HDD"]
y = df["Price_euros"]

# Add a constant term to the predictor variable to fit
# an intercept
X = sm.add_constant(X)

# Fit the linear regression model
model = sm.OLS(y, X).fit()

# Print the model summary
print(model.summary())

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.15, random_state=3
)

model = LinearRegression()
model.fit(X_train, y_train)


y_pred = model.predict(X_test)

accuracies = cross_val_score(estimator=model,
                             X=X_train, y=y_train, cv=5)

print(f"R2 Score : {r2_score(y_test,y_pred) * 100:.2f}%")
print(f"MAE : {mean_absolute_error(y_test,y_pred) * 100:.2f}%")
print(f"MSE : {mean_squared_error(y_test,y_pred) * 100:.2f}%")
print("Cross Val Accuracy: {:.2f} %".format(accuracies.mean()
                                            * 100))

model_comparison["SLR Vs HDD"] = [
    r2_score(y_test, y_pred),
    mean_squared_error(y_test, y_pred),
    mean_absolute_error(y_test, y_pred),
    (accuracies.mean()),
]


# ###Multilinear Regression

# ####Model 1 : with all the selected variables

# In[104]:


X = df.drop(columns=["Price_euros"])
y = np.log(df["Price_euros"])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.15, random_state=3
)

ct = ColumnTransformer(
    transformers=[
        (
            "onehot",
            OneHotEncoder(sparse=False, handle_unknown="ignore"),
            [0, 1, 3, 8, 11],
        )
    ],
    remainder="passthrough",
)

pipe = Pipeline(steps=[("preprocessor", ct), ("regressor",
                                              LinearRegression())])

pipe.fit(X_train, y_train)

y_pred = pipe.predict(X_test)

r2 = r2_score(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
accuracy = cross_val_score(estimator=pipe, X=X_train,
                           y=y_train, cv=5).mean()

print(f"R2 Score: {r2*100:.2f}%")
print(f"MSE: {mse*100:.2f}%")
print(f"MAE: {mae*100:.2f}%")
print(f"Cross Val Accuracy: {accuracy*100:.2f}%")

model_comparison["MLR(final)"] = [r2, mse, mae, accuracy]


# #### Model 2 : Except Weight
#

# In[105]:


X = df.drop(columns=["Price_euros", "Company"])
y = np.log(df["Price_euros"])
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.15, random_state=3
)
X_train


# In[107]:


ct = ColumnTransformer(
    transformers=[
        ("onehot", OneHotEncoder(sparse=False,
                                 handle_unknown="ignore"),
         [0, 2, 7, 10])
    ],
    remainder="passthrough",
)

pipe = Pipeline(steps=[("preprocessor", ct),
                       ("regressor", LinearRegression())])

pipe.fit(X_train, y_train)

y_pred = pipe.predict(X_test)

r2 = r2_score(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
accuracy = cross_val_score(estimator=pipe,
                           X=X_train, y=y_train, cv=5).mean()

print(f"R2 Score: {r2*100:.2f}%")
print(f"MSE: {mse*100:.2f}%")
print(f"MAE: {mae*100:.2f}%")
print(f"Cross Val Accuracy: {accuracy*100:.2f}%")

model_comparison["MLR w/o Weight"] = [r2, mse,
                                      mae, accuracy]


# #### Model 3: Except HDD
#

# In[108]:


X = df.drop(columns=["Price_euros", "HDD"])
y = np.log(df["Price_euros"])
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.15, random_state=3
)
X_train


# In[109]:


ct = ColumnTransformer(
    transformers=[
        (
            "onehot",
            OneHotEncoder(sparse=False, handle_unknown="ignore"),
            [0, 1, 3, 8, 10],
        )
    ],
    remainder="passthrough",
)

pipe = Pipeline(steps=[("preprocessor", ct), ("regressor",
                                              LinearRegression())])

pipe.fit(X_train, y_train)

y_pred = pipe.predict(X_test)

r2 = r2_score(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
accuracy = cross_val_score(estimator=pipe,
                           X=X_train, y=y_train, cv=5).mean()

print(f"R2 Score: {r2*100:.2f}%")
print(f"MSE: {mse*100:.2f}%")
print(f"MAE: {mae*100:.2f}%")
print(f"Cross Val Accuracy: {accuracy*100:.2f}%")

model_comparison["MLR w/o HDD"] = [r2, mse,
                                   mae, accuracy]


# ###Summary on Model Building
#
# 1. Models were built using both Simple Linear regression
# and Multi Linear regression as well and a comparison of
# those models are present in the table below.
# 2. Before starting to create a model, we had been choosing
# the varaibles based on various factors mostly based on the
# Analysis section, thus making all the selected variables to
# be most useful to build a model.
# 3. Despite the facts like the high multicollinearity of Ram
# and Weight against Price variable or getting a slightly higher a
# ccuracy in 3rd model of MultiLinear regression than the final model,
# trying to choose almost all useful and feature selected variables
# shall make the model more mature in certain terms and also we need
# to value the efforts put in to collect these useful data and try to
# handle and use them in the most effective and efficient ways.

# In[110]:


df_model_comparison = pd.DataFrame.from_dict(
    model_comparison,
    orient="index",
    columns=["R2 Score", "MSE", "MAE", "Cross Val Accuracy"],
)
df_model_comparison


# In[111]:


X = df.drop(columns=["Price_euros"])
y = np.log(df["Price_euros"])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.15, random_state=3
)

ct = ColumnTransformer(
    transformers=[
        (
            "onehot",
            OneHotEncoder(sparse=False,
                          handle_unknown="ignore"),
            [0, 1, 3, 8, 11],
        )
    ],
    remainder="passthrough",
)

pipe = Pipeline(steps=[("preprocessor", ct), ("regressor",
                                              LinearRegression())])

pipe.fit(X_train, y_train)

# R-squared and Adjusted R-squared values
print("Train R-squared:", pipe.score(X_train, y_train))
print("Test R-squared:", pipe.score(X_test, y_test))
n = X_train.shape[0]
p = X_train.shape[1]
train_adj_r2 = 1 - (1 - pipe.score(X_train, y_train)) * \
    (n - 1) / (n - p - 1)
print("Train Adjusted R-squared:", train_adj_r2)
n = X_test.shape[0]
p = X_test.shape[1]
test_adj_r2 = 1 - (1 - pipe.score(X_test, y_test)) * \
              (n - 1) / (n - p - 1)
print("Test Adjusted R-squared:", test_adj_r2)

# Performance comparison on Train and Test Samples
y_train_pred = pipe.predict(X_train)
y_test_pred = pipe.predict(X_test)
print("Train MAE:", mean_absolute_error(y_train, y_train_pred))
print("Test MAE:", mean_absolute_error(y_test, y_test_pred))
print("Train MSE:", mean_squared_error(y_train, y_train_pred))
print("Test MSE:", mean_squared_error(y_test, y_test_pred))
print("Train RMSE:", np.sqrt(mean_squared_error(y_train, y_train_pred)))
print("Test RMSE:", np.sqrt(mean_squared_error(y_test, y_test_pred)))

# Histogram of residual deviance on test data
residuals = y_test - y_test_pred
plt.hist(residuals, bins=20)
plt.xlabel("Residuals")
plt.ylabel("Frequency")
plt.title("Histogram of Residuals")
plt.show()

df.to_csv("df.csv", index=False)
pickle.dump(pipe, open("pipe.pkl", "wb"))
