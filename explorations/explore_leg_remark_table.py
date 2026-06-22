# Databricks notebook source
# MAGIC %sql
# MAGIC select leg_no, usage, count(*) 
# MAGIC from panda_silver_prod.occ_ops.netline___schedops__leg_remark
# MAGIC where usage = 'F'
# MAGIC group by all
# MAGIC having count(*) > 1
