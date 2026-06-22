# Databricks notebook source
from mlflow import MlflowClient
import mlflow
mlflow.set_registry_uri("databricks-uc")
MlflowClient().set_registered_model_alias(
       "panda_gold_dev.ml_ops.flight_delay_model", "champion", "9"
   )