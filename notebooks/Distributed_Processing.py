# Databricks notebook source
from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import LogisticRegression

# COMMAND ----------

# create a SparkSession
spark = SparkSession.builder.appName("my_app").getOrCreate()

# COMMAND ----------

# load a large CSV file using PySpark
df = spark.read.csv("path/to/large_dataset.csv", header=True, inferSchema=True)

# COMMAND ----------

# perform exploratory data analysis
df.printSchema()
df.describe().show()
df.select("label").groupBy("label").count().show()

# COMMAND ----------


# preprocess the data
feature_cols = ["feature1", "feature2", "feature3"]
assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")
df = assembler.transform(df)

# COMMAND ----------

# split the data into training and testing sets
train_data, test_data = df.randomSplit([0.7, 0.3], seed=42)

# COMMAND ----------

# train a logistic regression model
lr = LogisticRegression(featuresCol="features", labelCol="label", maxIter=10)
lr_model = lr.fit(train_data)

# COMMAND ----------

# evaluate the model on the test data
predictions = lr_model.transform(test_data)
predictions.select("label", "prediction").show()

# COMMAND ----------

# save the model
lr_model.save("path/to/model")
