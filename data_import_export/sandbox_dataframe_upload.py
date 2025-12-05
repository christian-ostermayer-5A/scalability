from databricks.sdk.runtime import *

import json

from hive_metastore_client import HiveMetastoreClient
from quintoandar_gsheets_api_client.clients import GoogleSheetsClient

from bietlejuice.base.api.api_enum import APIEnum
from bietlejuice.base.db.database_enum import DatabaseEnum
from bietlejuice.base.hive.table_storage_descriptor_enum import (
    TableStorageDescriptorEnum,
)
from bietlejuice.base.spark.spark_metastore_helper import SparkMetastoreHelper
from bietlejuice.base.spark.spark_table_storage_format import SparkTableStorageFormat
from bietlejuice.clients.db_clients.spark_client import SparkClient
from bietlejuice.consumers.api_consumers.gsheets_consumer import GsheetsConsumer
from bietlejuice.loaders.hive_metastore_loader import HiveMetastoreLoader
from bietlejuice.loaders.s3_loader import S3Loader
from bietlejuice.loaders.spark_metastore_loader import SparkMetastoreLoader
from bietlejuice.services.metastore_services.hive_metastore_service import (
    HiveMetastoreService,
)
from bietlejuice.services.metastore_services.spark_metastore_service import (
    SparkMetastoreService,
)

import os
import re


def sandbox_dataframe_upload(df, 
                             table_name,
                             TIMEOUT_LIMIT = 300):

  assert re.match(r'^[a-zA-Z][\w\-|]+$|^$', table_name), "Invalid table name parameter: must start with a letter and use only letters (upper or lower cases), numbers, hyphens or underscores."


  environment="prod"
  sandbox_bucket="5a-sandbox-prod"
  database_name = "sandbox"
  layer = "clean"
  database_location = f"s3a://{sandbox_bucket}/{layer}/" 
  all_tables_flag=False
  partition_keys=[]

  credentials=json.loads(dbutils.secrets.get(scope="quintoandar", key=APIEnum.GSHEETS_CREDENTIALS))
  scope = credentials.pop("scope")
  gsheets_client = GoogleSheetsClient(credentials, scope, timeout=TIMEOUT_LIMIT)
  spark_client = SparkClient()

  gsheets_consumer = GsheetsConsumer(gsheets_client, spark_client)

  # Importar SparkSession se não estiver disponível

  from pyspark.sql import SparkSession
  # Criar uma SparkSession se não existir
  spark = SparkSession.builder.getOrCreate()
  # Converter o Pandas DataFrame para Spark DataFrame
  spark_df = df#spark.createDataFrame(df)

  storage_file_format = SparkTableStorageFormat.get_storage(layer)
  s3_loader = S3Loader()
  s3_loader.load_df(
      df=spark_df,
      s3_path=f"{database_location}/{table_name}",
      format_options=storage_file_format,
  )

  # Carregar o DataFrame no S3 usando o Spark DataFrame

  spark_metastore_service = SparkMetastoreService(spark_client)
  spark_metastore_loader = SparkMetastoreLoader(spark_metastore_service)
  spark_metastore_service.create_database(database_name)
  spark_metastore_loader.update_metastore(
    spark_df, database_name, table_name, storage_file_format, database_location
  )

  hive_ms_host = json.loads(dbutils.secrets.get("quintoandar", DatabaseEnum.HIVE_METASTORE))["host"]
  hive_ms_client = HiveMetastoreClient(hive_ms_host)
  hive_ms_service = HiveMetastoreService(hive_ms_client)
  hive_ms_loader = HiveMetastoreLoader(hive_ms_service)
  storage_description = TableStorageDescriptorEnum.from_layer(layer)

  spark_ms = SparkMetastoreHelper(sandbox_bucket, layer, database_name, table_name, all_tables_flag)
  spark_ms.spark_database_name = database_name

  hive_ms_loader.hive_metastore_service.create_database(database_name)
  columns = spark_ms.get_spark_metastore_table_columns(table_name)
  hive_ms_loader.sync_metastore(
      database_name=database_name,
      table_name=table_name,
      database_location=database_location,
      table_schema=columns,
      partition_keys=partition_keys,
      format_info=storage_description,
      source_schema=columns,
  )
