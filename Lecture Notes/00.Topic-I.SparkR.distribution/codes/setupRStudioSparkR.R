#
# Licensed to the Apache Software Foundation (ASF) under one or more
# contributor license agreements.  See the NOTICE file distributed with
# this work for additional information regarding copyright ownership.
# The ASF licenses this file to You under the Apache License, Version 2.0
# (the "License"); you may not use this file except in compliance with
# the License.  You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

#-------------------------------------
# STEP 1: Set System Environment
# SPARK_HOME variable using 
# the path to Spark installation folder
#--------------------------------------
Sys.setenv(SPARK_HOME="E:/Samatova_EXT/0-LECTURES/2-NCSU-BDA/Spark/spark-1.5.0-bin-hadoop2.6")


#---------------------------------------
# STEP 2: Set the library path for Spark 
#---------------------------------------
.libPaths(c(file.path(Sys.getenv("SPARK_HOME"),
            "R","lib"),
            .libPaths()))

#---------------------------------------
# STEP 3: Load SparkR library 
#---------------------------------------
library(SparkR)

#---------------------------------------
# STEP 4: Initialize SparkContext
#     The argument in this command is 
#     master = "local[N]", where N stands for 
#     the number of threads to use.
#---------------------------------------
sc <- sparkR.init(master="local")

#---------------------------------------
# STEP 4: Initialize SQLContext
#         from your Spark context 
#---------------------------------------
sqlContext <- sparkRSQL.init(sc)

