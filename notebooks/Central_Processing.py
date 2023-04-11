# Databricks notebook source
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

# COMMAND ----------


# load the dataset using pandas
df = pd.read_csv("path/to/large_dataset.csv")

# COMMAND ----------

# perform exploratory data analysis
print(df.info())
print(df.describe())
print(df["label"].value_counts())

# COMMAND ----------

# preprocess the data
feature_cols = ["feature1", "feature2", "feature3"]
scaler = StandardScaler()
X = scaler.fit_transform(df[feature_cols])
y = df["label"]

# COMMAND ----------

# split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# COMMAND ----------

# train a logistic regression model
lr = LogisticRegression(max_iter=10)
lr.fit(X_train, y_train)

# COMMAND ----------

# evaluate the model on the test data
y_pred = lr.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy: ", accuracy)

# COMMAND ----------

# save the model
with open("path/to/model.pkl", "wb") as f:
    pickle.dump(lr, f)
