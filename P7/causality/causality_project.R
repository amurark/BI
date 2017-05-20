# Load the libraries 
#To install pcalg library you may first need to execute the following commands:
# source("https://bioconductor.org/biocLite.R")
biocLite("graph")
biocLite("RBGL")
biocLite("Rgraphviz")
#install.packages("pcalg")
#install.packages("vars")
#install.packages("urca")
#install.packages("Rgraphviz")
#library(stats)
library(graph)
library(RBGL)
library(Rgraphviz)
library(vars)
library(pcalg)
library(urca)

# Read the input data 
data <- read.csv(file="data.csv",head=TRUE,sep=",")

# Build a VAR model 
# Select the lag order using the Schwarz Information Criterion with a maximum lag of 10
# see ?VARSelect to find the optimal number of lags and use it as input to VAR()
VARselect(data, lag.max = 10, type = c("const"), season = NULL, exogen = NULL)
varModel <- VAR(data, p = 1)

# Extract the residuals from the VAR model 
# see ?residuals
modelRes <- residuals(varModel)


# Check for stationarity using the Augmented Dickey-Fuller test 
# see ?ur.df
Move <- ur.df(modelRes[,1])
RPRICE <- ur.df(modelRes[,2])
MPRICE <- ur.df(modelRes[,3])

#Since the p-value is lower than significance value(0.05) 
#We can reject the NULL-HYPOTHESIS and hence we can conclude that its stationary
summary(Move)
summary(RPRICE)
summary(MPRICE)

# Check whether the variables follow a Gaussian distribution  
# see ?ks.test
#Since p-value is less than significance value, it doesn't follow gaussian distribution.
ks.test(modelRes[,1], 'pnorm')
ks.test(modelRes[,2], 'pnorm')
ks.test(modelRes[,3], 'pnorm')

# Write the residuals to a csv file to build causal graphs using Tetrad software
write.csv(modelRes, file = "residuals.csv", row.names = FALSE)

# OR Run the PC and LiNGAM algorithm in R as follows,
# see ?pc and ?LINGAM


# PC Algorithm
pc.fit <- pc(suffStat = list(C=cor(modelRes), n=1000), indepTest=gaussCItest, alpha=0.1, labels=colnames(data), skel.method="original")
plot(pc.fit, main = "Causal Graph using PC")

# LiNGAM Algorithm
dag <- lingam(modelRes, verbose = TRUE)
show(dag)
plot(dag$Bpruned)
