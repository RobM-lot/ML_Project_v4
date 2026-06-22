# Databricks notebook source
spark.table("panda_silver_dev.ml_ops.ft_leg_status").select("event_ts").limit(1).show()