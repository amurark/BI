require(fpp) 
data(pigs)
fit <- snaive(pigs)
res <- residuals(fit)
Acf(res)
test <- Box.test(res, lag=1, type = "Lj")
if(test["p.value"] <= 0.05){
  print("Not a white noise")
}else{
  print("White Noise")
}
