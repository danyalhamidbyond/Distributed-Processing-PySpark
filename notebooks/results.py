# Databricks notebook source
import pandas as pd 

# COMMAND ----------

df_time = pd.read_csv('/dbfs/FileStore/shared_uploads/danyal.hamid@b-yond.com/learning_output/time_dist.csv')
df_d= df_time 
df_d = df_d.drop(df_d.index[0])
df_d = df_d.drop(df_d.index[4])

# COMMAND ----------

df_time = pd.read_csv('/dbfs/FileStore/shared_uploads/danyal.hamid@b-yond.com/learning_output/time_central.csv')
df_c = df_time

# COMMAND ----------

import matplotlib.pyplot as plt

# Create subplots
fig, axs = plt.subplots(nrows=3, ncols=3, figsize=(12, 8))
fig.tight_layout(pad=2)
# Loop through each column and plot against num files
for i, col in enumerate(df_c.columns):
    if col != 'number of rows':
        if col != "num files":
            row = i // 3
            col1 = i % 3
            print(row,col)
            axs[row, col1].plot(df_c['number of rows']/1000000, df_c[col]/1000)
            axs[row, col1].plot(df_d['number of rows']/1000000, df_d[col]/1000)
            axs[row, col1].set_xlabel('number of rows (mil)')
            axs[row, col1].set_ylabel(col,fontweight='bold')
            axs[2, 2].axis('off')

            

# COMMAND ----------

import matplotlib.pyplot as plt
# Create subplots
fig, axs = plt.subplots(nrows=3, ncols=3, figsize=(12, 8))
fig.tight_layout(pad=5)
# Loop through each column and plot against num files
for i, col in enumerate(df_d.columns):
    if col != 'number of rows':
        if col != "num files":
            row = i // 3
            col1 = i % 3
            #axs[row, col1].plot(df_c['number of rows'], df_c[col]/1000)
            axs[row, col1].plot(df_d['number of rows']/1000000, df_d[col]/1000)
            axs[row, col1].set_xlabel('number of rows (mil)')
            axs[row, col1].set_ylabel(col,fontweight='bold')
            axs[2, 2].axis('off')


# COMMAND ----------


