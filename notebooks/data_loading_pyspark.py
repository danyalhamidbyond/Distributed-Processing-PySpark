# Databricks notebook source
from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import LogisticRegression

# create a SparkSession
spark = SparkSession.builder.appName("my_app").getOrCreate()

# COMMAND ----------

import time
list_time = []

for i in range(8):
    num_files = i
    ####################################################################################################################################################################################################################
    from pyspark.sql.functions import input_file_name
    import os
    import zipfile

    # Define the directory path
    dir_path = "/dbfs/FileStore/shared_uploads/danyal.hamid@b-yond.com/reviews"

    # Read the schema of the first file in the directory
    first_file_path = os.path.join(dir_path, os.listdir(dir_path)[1])
    print(first_file_path)

    first_df = (
        spark.read.option("delimiter", "\t")
        .option("inferSchema", "true")
        .option("header", "true")
        .csv(first_file_path[5:])
    )
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
            df = (
                spark.read.option("delimiter", "\t")
                .schema(schema)
                .option("header", "true")
                .csv(file_path[5:])
            )
            # Append the data to the combined DataFrame
            combined_df = combined_df.union(df)
            count = count + 1
    # Show the combined DataFrame

    combined_df.show()
    ####################################################################################################################################################################################################################
    end_time = time.time()
    # calculate execution time in milliseconds
    execution_time_ms = (end_time - start_time) * 1000
    list_time.append(execution_time_ms)
    print("Execution time: ", execution_time_ms, "ms")
print(list_time)

# COMMAND ----------

import matplotlib.pyplot as plt

# x and y values as lists
x = list(range(1, 9))

# create a line plot
plt.plot(x, list_time)

# add axis labels and title
plt.xlabel('Number of files and overall size')
plt.ylabel('time taken')
plt.title('First run for data loading')

# show the plot
plt.show()

# COMMAND ----------

import matplotlib.pyplot as plt

# x and y values as lists
x = list(range(1, 9))

# create a line plot
plt.plot(x, list_time)

# add axis labels and title
plt.xlabel('Number of files and overall size')
plt.ylabel('time taken')
plt.title('second run for data loading')

# show the plot
plt.show()

# COMMAND ----------

import matplotlib.pyplot as plt

# x and y values as lists
x = list(range(1, 9))

# create a line plot
plt.plot(x, list_time)

# add axis labels and title
plt.xlabel('Number of files and overall size')
plt.ylabel('time taken')
plt.title('Third run for data loading')

# show the plot
plt.show()

# COMMAND ----------

import matplotlib.pyplot as plt

# x and y values as lists
x = list(range(1, 9))

# create a line plot
plt.plot(x, list_time)

# add axis labels and title
plt.xlabel('Number of files and overall size')
plt.ylabel('time taken')
plt.title('fourth run for data loading')

# show the plot
plt.show()

# COMMAND ----------

import matplotlib.pyplot as plt

# x and y values as lists
x = list(range(1, 9))

# create a line plot
plt.plot(x, list_time)

# add axis labels and title
plt.xlabel('Number of files and overall size')
plt.ylabel('time taken')
plt.title('fifth run for data loading')

# show the plot
plt.show()

# COMMAND ----------


