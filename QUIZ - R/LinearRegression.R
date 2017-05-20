states <- as.data.frame(
  state.x77[,c("Murder","Population",
               "Illiteracy", "Income", "Frost")])
dim(states)
t(states[1,])
dtrain <- states[1:25,]
dtest <- states[26:50,]
murderModel <- lm (Murder ~ Population + Illiteracy 
                   + Income + Frost, data=dtrain)
summary (murderModel) 



####Independence####
#The rows are independent of each other.

###Linearity
library(car)
murderModel <- lm (Murder ~ sqrt(Population) + Illiteracy 
                   + Income + Frost, data=dtrain)

crPlots(murderModel)


###Normality
qqnorm(states$Murder)
library(gvlma)
gvmodel <- gvlma(murderModel) 
summary(gvmodel)
##Since the graph from qqnorm is more or less linear in nature, the residuals in this model are normally distributed.
##Also, gvlma performs a global validation of linear model assumptions as well separate evaluations of skewness, kurtosis, and heteroscedasticity


###Error/Noise
durbinWatsonTest(murderModel)
#The null hypothesis is that there is no corelation between the residuals
##As p value is close to zero the null hypothesis can be rejected and there is no evidence of corelated errors.

##Homoscedasticity
ncvTest(murderModel)
##Since the value of p > 0.05, we can infer that the model shows heteroscedasticity is absent.


####Multicollinearity
vif(murderModel)
##Since the variance inflation factors are small, the models show an absence of multicollinearity

####Sensitivity to outliers
outlierTest(murderModel)
##The p value is greater than 0.05. Hence the model is sensitive to outliers.

####Model Complexity
##The coefficient as shown in the summary show that the model used does not have a high complexity.
