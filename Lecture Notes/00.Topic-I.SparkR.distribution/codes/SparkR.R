#----------------------Setting up SparkR-----------------------

#Set SPARK_HOME to SPARK installation folder
spark_home <- "/home/anu/Spark_1.5/spark-1.5.0-bin-hadoop2.6"
Sys.setenv(SPARK_HOME=spark_home) 

#point to sparkR library in the installation
.libPaths(c(file.path(Sys.getenv("SPARK_HOME"), "R", "lib"), .libPaths()))  
library(SparkR)    #Include the sparkR library

#Creating sparkContext and wraps a sqlContext around it
sc <- sparkR.init()
sqlContext <- sparkRSQL.init(sc)


#--------------------SparkR dataframes - Support for various data formats-----------------------


#reading data from a json file into a data frame
people <- read.df(sqlContext, paste(spark_home,"/examples/src/main/resources/people.json", sep=""), "json")
head(people)
# SparkR automatically infers the schema from the JSON file
printSchema(people)

#Writing to a parquet file
write.df(people, path="people.parquet", source="parquet", mode="overwrite") 


#--------------------Selecting rows, columns-----------------------


# Create a SparkR DataFrame
df <- createDataFrame(sqlContext, faithful) 
df
#Select only the "eruptions" column
head(select(df, df$eruptions))
# You can also pass in column name as strings
head(select(df, "eruptions"))

# Filter the DataFrame to only retain rows with wait times shorter than 50 mins
head(filter(df, df$waiting < 50))


#--------------------Grouping, Aggregations----------------------


#We use the `n` operator to count the number of times each waiting time appears
head(summarize(groupBy(df, df$waiting), count = n(df$waiting)))

# We can also sort the output from the aggregation to get the most common waiting times
waiting_counts <- summarize(groupBy(df, df$waiting), count = n(df$waiting))
head(arrange(waiting_counts, desc(waiting_counts$count)))


#--------------------Run SQL queries from SparkR----------------------


# Load a JSON file
people <- read.df(sqlContext, paste(spark_home,"/examples/src/main/resources/people.json", sep=""), "json")

# Register this DataFrame as a table.
registerTempTable(people, "people")

# SQL statements can be run by using the sql method
teenagers <- sql(sqlContext, "SELECT name FROM people WHERE age >= 13 AND age <= 19")
head(teenagers)


#--------------------SparkR and Machine Learning----------------------


# Create the DataFrame
df <- createDataFrame(sqlContext, iris)

# Fit a linear model over the dataset.
model <- glm(Sepal_Length ~ Sepal_Width + Species, data = df, family = "gaussian")

# Model coefficients are returned in a similar format to R's native glm().
summary(model)

# Make predictions based on the model.
predictions <- predict(model, newData = df)
head(select(predictions, "Sepal_Length", "prediction"))
