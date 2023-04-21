# Databricks notebook source
from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import LogisticRegression

# COMMAND ----------

# create a SparkSession
spark = SparkSession.builder.appName("my_app").getOrCreate()

# COMMAND ----------

# MAGIC %md
# MAGIC #Load data

# COMMAND ----------

import time
list_time = []
num_files = 8

# COMMAND ----------

from pyspark.sql.functions import input_file_name
import os

# Define the directory path
dir_path = "/dbfs/FileStore/shared_uploads/danyal.hamid@b-yond.com/reviews"

# Read the schema of the first file in the directory
first_file_path = os.path.join(dir_path, os.listdir(dir_path)[1])
print(first_file_path)


first_df = spark.read.option("delimiter", "\t").option("inferSchema", "true").option("header", "true").csv(first_file_path[5:])
schema = first_df.schema
print(first_df.columns)

# Create an empty DataFrame to hold the combined data
combined_df = spark.createDataFrame([], schema)
count = 0 

start_time = time.time()
# Loop through the list of files in the directory and read each TSV file into a DataFrame
for file_name in os.listdir(dir_path):
    if count < num_files + 1:
        file_path = os.path.join(dir_path, file_name)
        print(file_path)
        # Read the unzipped file as a CSV file
        df = spark.read.option("delimiter", "\t").schema(schema).option("header", "true").csv(file_path[5:])
        # Append the data to the combined DataFrame
        combined_df = combined_df.union(df)
        count = count + 1
# Show the combined DataFrame
combined_df.show()

end_time = time.time()
# calculate execution time in milliseconds
execution_time_ms = (end_time - start_time) * 1000
list_time.append(execution_time_ms)

# COMMAND ----------

# Get the number of rows and columns in the DataFrame
num_rows = combined_df.count()
num_cols = len(combined_df.columns)


# COMMAND ----------

print("Shape of DataFrame: ({}, {})".format(num_rows, num_cols))

# COMMAND ----------

start_time = time.time()

# Import necessary libraries
from pyspark.sql.functions import col

# Print schema of the DataFrame
df.printSchema()

# Descriptive statistics of the DataFrame
df.describe().show()

# Value counts of the "star_rating" column
df.groupBy("star_rating").count().orderBy(col("count").desc()).show()

end_time = time.time()
# calculate execution time in milliseconds
execution_time_ms = (end_time - start_time) * 1000
list_time.append(execution_time_ms)

# COMMAND ----------

start_time = time.time()

# Import necessary libraries
from pyspark.sql.functions import udf
from pyspark.sql.types import IntegerType

# Define a UDF to count words in a row
count_words_udf = udf(lambda row: len(str(row).split()), IntegerType())

# Apply the UDF to the desired column
df = df.withColumn('count_review_body', count_words_udf(df['review_body']))

# Display the DataFrame
df.show(5)

end_time = time.time()
# calculate execution time in milliseconds
execution_time_ms = (end_time - start_time) * 1000
list_time.append(execution_time_ms)

# COMMAND ----------

# Define a UDF to count words in a row
count_words_udf = udf(lambda row: len(str(row).split()), IntegerType())

# Apply the UDF to the desired column
df = df.withColumn("count_headline_body", count_words_udf(df["review_headline"]))

# Display the DataFrame
df.show(5)

# COMMAND ----------

start_time = time.time()

# Import necessary libraries
from pyspark.sql.functions import unix_timestamp
from pyspark.sql.types import DoubleType

# Convert date column to Unix timestamps
df = df.withColumn('timestamp', unix_timestamp(df['review_date'], 'yyyy-MM-dd').cast(DoubleType()))

# Display the updated DataFrame
df.show()

end_time = time.time()
# calculate execution time in milliseconds
execution_time_ms = (end_time - start_time) * 1000
list_time.append(execution_time_ms)

# COMMAND ----------

start_time = time.time()

# Drop all rows that contain any null or missing values
df = df.dropna()

end_time = time.time()
# calculate execution time in milliseconds
execution_time_ms = (end_time - start_time) * 1000
list_time.append(execution_time_ms)

# COMMAND ----------

# Select specific columns from the DataFrame
y = df.select("star_rating")
df1 = df.select(
    "star_rating",
    "helpful_votes",
    "total_votes",
    "vine",
    "verified_purchase",
    "count_review_body",
    "count_headline_body",
    "timestamp",
)

# Display the updated DataFrame
df1.show(5)

# COMMAND ----------

df.select("vine").distinct().show()

# COMMAND ----------

start_time = time.time()

from pyspark.ml.feature import OneHotEncoder
from pyspark.ml import Pipeline
from pyspark.ml.feature import StringIndexer

# Define columns to encode
columns_to_encode = ['vine', 'verified_purchase']

# Create a pipeline to encode the columns
pipeline = Pipeline(stages=[StringIndexer(inputCol=c, outputCol=c+"_index") for c in columns_to_encode] +
                           [OneHotEncoder(inputCols=[c+"_index" for c in columns_to_encode],
                                          outputCols=[c+"_encoded" for c in columns_to_encode])])

# Fit and transform the DataFrame using the pipeline
df_encoded = pipeline.fit(df1).transform(df1)

# Display the updated DataFrame
df_encoded.show(5)

end_time = time.time()
# calculate execution time in milliseconds
execution_time_ms = (end_time - start_time) * 1000
list_time.append(execution_time_ms)

# COMMAND ----------

X = df_encoded.select(
    "helpful_votes",
    "total_votes",
    "vine_index",
    "verified_purchase_index",
    "count_review_body",
    "count_headline_body",
    "timestamp",
)
X.show(5)
y.show(5)

df_final = df_encoded.select(
    "star_rating",
    "helpful_votes",
    "total_votes",
    "vine_index",
    "verified_purchase_index",
    "count_review_body",
    "count_headline_body",
    "timestamp",
)
# preprocess the data
feature_cols = [
    "helpful_votes",
    "total_votes",
    "vine_index",
    "verified_purchase_index",
    "count_review_body",
    "count_headline_body",
    "timestamp",
]
assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")
df_final = assembler.transform(df_final)

# COMMAND ----------

start_time = time.time()

# split the data into training and testing sets
train_data, test_data = df_final.randomSplit([0.7, 0.3], seed=42)

end_time = time.time()
# calculate execution time in milliseconds
execution_time_ms = (end_time - start_time) * 1000
list_time.append(execution_time_ms)

# COMMAND ----------

start_time = time.time()

# Define the hyperparameters
hyperparams = {
    'numTrees': 10,
    'maxDepth': 5,
    'minInstancesPerNode': 2,
    'minInfoGain': 0.0
}

# Import necessary libraries
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator

# Train a PySpark random forest model
rf = RandomForestClassifier(featuresCol="features", labelCol='star_rating',
                             numTrees=hyperparams['numTrees'],
                             maxDepth=hyperparams['maxDepth'],
                             minInstancesPerNode=hyperparams['minInstancesPerNode'],
                             minInfoGain=hyperparams['minInfoGain'])
rf_model = rf.fit(train_data)

# Make predictions on the test data
predictions = rf_model.transform(test_data)

# Evaluate the model performance
evaluator = BinaryClassificationEvaluator(rawPredictionCol="prediction", labelCol='star_rating')
auc = evaluator.evaluate(predictions, {evaluator.metricName: "areaUnderROC"})
print("Test AUC:", auc) 

end_time = time.time()
# calculate execution time in milliseconds
execution_time_ms = (end_time - start_time) * 1000
list_time.append(execution_time_ms)

# COMMAND ----------

list_time.append(num_files)
list_time.append(num_rows)

# COMMAND ----------

import pandas as pd
import os

# Define the path to your CSV file
csv_path = '/dbfs/FileStore/shared_uploads/danyal.hamid@b-yond.com/learning_output/time_dist.csv'

# Define the new row as a list of values
new_row = list_time
# Check if the CSV file already exists
if os.path.isfile(csv_path):
    # Load the CSV file into a Pandas DataFrame
    df = pd.read_csv(csv_path)
    
    # Append the new row to the DataFrame
    df.loc[len(df)] = new_row
else:
    # Create a new DataFrame with the header row and the new row
    df = pd.DataFrame(columns=['load data', 'value counts', 'word count', 'date transform', 'drop na', 'one hor encoding', 'train test split', 'model training', 'num files', 'number of rows'])
    df.loc[0] = new_row
    
# Write the updated DataFrame to the CSV file
df.to_csv(csv_path, index=False)


# COMMAND ----------

df_time = pd.read_csv('/dbfs/FileStore/shared_uploads/danyal.hamid@b-yond.com/learning_output/time_dist.csv')
df_time

# COMMAND ----------


