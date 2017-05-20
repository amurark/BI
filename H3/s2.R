require(fpp)

#usnetelec
data("usnetelec")
plot(usnetelec)
diff <- ndiffs(usnetelec)
#The data is not seasonal
usnetelec_diff <- diff(usnetelec, lag=frequency(usnetelec), differences = diff)
plot(usnetelec_diff)
acf(usnetelec_diff)
diff_new <- ndiffs(usnetelec_diff)

#usgdp
data("usgdp")
plot(usgdp)
diff <- ndiffs(usgdp)
usg_diff <- diff(usgdp,  differences = diff)
plot(usg_diff)
acf(usg_diff)
diff_new <- ndiffs(usg_diff)


#mcopper
data("mcopper")
plot(mcopper)
diff <- ndiffs(mcopper)
mcopper_diff <- diff(mcopper)
plot(mcopper_diff)
acf(mcopper_diff)
diff2 <- ndiffs(mcopper_diff)


#enplanements
data("enplanements")
plot(enplanements)
diff <- ndiffs(enplanements)
#Seasonal Data
nsdiff <- nsdiffs(enplanements)
#Needs both mean and variance stabilization
l <- BoxCox.lambda(enplanements)
enplanements_box <- BoxCox(enplanements,l)
enplanements_diff <- diff(enplanements_box)
plot(enplanements_diff)
acf(enplanements_diff)
diff_new <- ndiffs(enplanements_diff)


#Visitors
data("visitors")
plot(visitors)
diff <- ndiffs(visitors)
#Seasonal Data
nsdiff <- nsdiffs(visitors)
#Needs both mean and variance stabilization
l <- BoxCox.lambda(visitors)
visitor_box <- BoxCox(visitors,l)
vis_diff <- diff(visitor_box)
plot(vis_diff)
acf(vis_diff)
diff_new <- ndiffs(vis_diff)

