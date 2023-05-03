# Databricks notebook source
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

# COMMAND ----------

import os
import pandas as pd
import zipfile

# Define the directory path
dir_path = "/dbfs/FileStore/shared_uploads/danyal.hamid@b-yond.com/reviews"

# Read the schema of the first file in the directory
first_file_path = os.path.join(dir_path, os.listdir(dir_path)[1])
print(first_file_path)
first_df = pd.read_csv(first_file_path, sep="\t", header=0, error_bad_lines=False)
columns = first_df.columns

# COMMAND ----------

import time
list_time = []
num_files = 1

# COMMAND ----------

start_time = time.time()


combined_df = pd.DataFrame(columns=columns)
count = 0 

# Loop through the list of files in the directory and read each TSV file into a DataFrame
for file_name in os.listdir(dir_path):
    if file_name != '1.tsv' and count < num_files:
        file_path = os.path.join(dir_path, file_name)
        print(file_path)

        # Read the unzipped file as a CSV file
        df = pd.read_csv(os.path.join(dir_path, file_name), sep="\t", header=0, error_bad_lines=False)
        
        # Append the data to the combined DataFrame
        combined_df = combined_df.append(df, ignore_index=True)
        count = count + 1

# Show the combined DataFrame
print(combined_df)
df = combined_df

end_time = time.time()
# calculate execution time in milliseconds
execution_time_ms = (end_time - start_time) * 1000
list_time.append(execution_time_ms)

# COMMAND ----------

# perform exploratory data analysis
start_time = time.time()

print(df.info())
print(df.describe())
print(df["star_rating"].value_counts())

end_time = time.time()
# calculate execution time in milliseconds
execution_time_ms = (end_time - start_time) * 1000
list_time.append(execution_time_ms)

# COMMAND ----------

start_time = time.time()

# define a lambda function to count words in a row
count_words = lambda row: len(str(row).split())

# apply the function to the desired column
df['count_review_body'] = df['review_body'].apply(count_words)

print(df)

end_time = time.time()
# calculate execution time in milliseconds
execution_time_ms = (end_time - start_time) * 1000
list_time.append(execution_time_ms)


# COMMAND ----------

# define a lambda function to count words in a row
count_words = lambda row: len(str(row).split())

# apply the function to the desired column
df["count_review_head"] = df["review_headline"].apply(count_words)

print(df)

# COMMAND ----------

start_time = time.time()

import datetime
import numpy as np

# Convert the date column to Unix timestamps
# df['timestamp'] = df['review_date'].apply(lambda x: int((datetime.datetime.strptime(str(x), '%Y-%m-%d') - datetime.datetime(1970, 1, 1)).total_seconds()))

# Convert the date column to Unix timestamps, handling missing and non-string values
df["timestamp"] = df["review_date"].apply(
    lambda x: int(
        (
            datetime.datetime.strptime(str(x), "%Y-%m-%d")
            - datetime.datetime(1970, 1, 1)
        ).total_seconds()
    )
    if isinstance(x, str)
    else np.nan
)

# Print the updated dataframe
print(df)


end_time = time.time()
# calculate execution time in milliseconds
execution_time_ms = (end_time - start_time) * 1000
list_time.append(execution_time_ms)

# COMMAND ----------

start_time = time.time()

df = df.dropna()

end_time = time.time()
# calculate execution time in milliseconds
execution_time_ms = (end_time - start_time) * 1000
list_time.append(execution_time_ms)

# COMMAND ----------

df1 = df[['star_rating','helpful_votes','total_votes','vine','verified_purchase','count_review_body','count_review_head','timestamp']]
display(df1.head(5))

# COMMAND ----------

start_time = time.time()

y = df1["star_rating"]

# One-hot encode columns color and size
df_encoded = pd.get_dummies(df1, columns=["vine", "verified_purchase"])

# Print the updated dataframe
# display(df_encoded)

end_time = time.time()
# calculate execution time in milliseconds
execution_time_ms = (end_time - start_time) * 1000
list_time.append(execution_time_ms)

# COMMAND ----------

X = df_encoded[['star_rating','helpful_votes','total_votes','vine_Y','verified_purchase_Y','count_review_body','count_review_head','timestamp']]

# COMMAND ----------

# split the data into training and testing sets
start_time = time.time()
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

end_time = time.time()
# calculate execution time in milliseconds
execution_time_ms = (end_time - start_time) * 1000
list_time.append(execution_time_ms)

# COMMAND ----------

y_train = y_train.astype(int)
y_test = y_test.astype(int)
y_train.unique()

# COMMAND ----------

start_time = time.time()
# Define the hyperparameters
hyperparams = {
    "numTrees": 10,
    "maxDepth": 5,
    "minInstancesPerNode": 2,
    "minInfoGain": 0.0,
}

# Train a scikit-learn random forest model
from sklearn.ensemble import RandomForestClassifier

rfc = RandomForestClassifier(
    n_estimators=hyperparams["numTrees"],
    max_depth=hyperparams["maxDepth"],
    min_samples_split=hyperparams["minInstancesPerNode"],
    min_impurity_decrease=hyperparams["minInfoGain"],
)
rfc.fit(X_train, y_train)


# Evaluate the model on the test data
score = rfc.score(X_test, y_test)
print("Accuracy:", score)


end_time = time.time()
# calculate execution time in milliseconds
execution_time_ms = (end_time - start_time) * 1000
list_time.append(execution_time_ms)

# COMMAND ----------

print(list_time)

# COMMAND ----------

list_time.append(num_files)
list_time.append(len(df))

# COMMAND ----------

import pandas as pd

# Load the CSV file into a Pandas dataframe
df = pd.read_csv('/dbfs/FileStore/shared_uploads/danyal.hamid@b-yond.com/learning_output/time_central.csv')

# Add a new row to the dataframe
new_row = list_time

# Append the new row to the DataFrame
df.loc[len(df)] = new_row

# Write the updated DataFrame back to the CSV file
df.to_csv('/dbfs/FileStore/shared_uploads/danyal.hamid@b-yond.com/learning_output/time_central.csv', index=False)


# COMMAND ----------

import pandas as pd
df_time = pd.read_csv('/dbfs/FileStore/shared_uploads/danyal.hamid@b-yond.com/learning_output/time_central.csv')
df_time

# COMMAND ----------


