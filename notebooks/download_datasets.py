# Databricks notebook source
!pip install kaggle 

# COMMAND ----------

# MAGIC %sh
# MAGIC cd ~/.kaggle
# MAGIC cp /dbfs/FileStore/shared_uploads/danyal.hamid@b-yond.com/kaggle.json ~/.kaggle/
# MAGIC chmod 600 ~/.kaggle/kaggle.json

# COMMAND ----------

!kaggle datasets download -d cynthiarempel/amazon-us-customer-reviews-dataset -f amazon_reviews_us_Music_v1_00.tsv -p /dbfs/FileStore/shared_uploads/danyal.hamid@b-yond.com/reviews

# COMMAND ----------


