df <- read.table("eBayAuctions.csv",header = TRUE,sep = ",", stringsAsFactors = F)
install.packages("dplyr")
library(dplyr)
data_tdf <- tbl_df(df)

smp_size <- floor(0.60 * nrow(data_tdf))

## set the seed to make the partition reproducible
set.seed(123)
train_ind <- sample(seq_len(nrow(data_tdf)), size = smp_size)

train <- data_tdf[train_ind, ]
test <- data_tdf[-train_ind, ]


cat_dtf <- summarise(group_by(train, Category), Competitive.=sum(Competitive.)/length(Competitive.))
cur_dtf <- summarise(group_by(train, currency), Competitive.=sum(Competitive.)/length(Competitive.))
end_dtf <- summarise(group_by(train, endDay), Competitive.=sum(Competitive.)/length(Competitive.))

#factor(train$Category, levels = c("Antique/Art/Craft", "Books", "Clothing/Accessories", "Toys/Hobbies", "Automotive", "Pottery/Glass", "Business/Industrial", "Electronics", "Coins/Stamps", "Jewelry", "Collectibles", "Music/Movie/Game", "Computer", "Home/Garden", "SportingGoods", "EverythingElse", "Health/Beauty", "Photography"), labels = c("ant_books_clot_toy", "ant_books_clot_toy", "ant_books_clot_toy", "ant_books_clot_toy","aut_pot","aut_pot","bus_elec", "bus_elec","coins_jewel", "coins_jewel","coll_music", "coll_music", "comp_home_sport","comp_home_sport","comp_home_sport", "everythingelse", "health", "photo"))


train$Category[train$Category == 'Antique/Art/Craft'] <- 'ant_books_clot_toy'
train$Category[train$Category == 'Books'] <- 'ant_books_clot_toy'
train$Category[train$Category == 'Clothing/Accessories'] <- 'ant_books_clot_toy'
train$Category[train$Category == 'Toys/Hobbies'] <- 'ant_books_clot_toy'
train$Category[train$Category == 'Automotive'] <- 'aut_pot'
train$Category[train$Category == 'Pottery/Glass'] <- 'aut_pot'
train$Category[train$Category == 'Business/Industrial'] <- 'bus_elec'
train$Category[train$Category == 'Electronics'] <- 'bus_elec'
train$Category[train$Category == 'Coins/Stamps'] <- 'coins_jewel'
train$Category[train$Category == 'Jewelry'] <- 'coins_jewel'
train$Category[train$Category == 'Collectibles'] <- 'coll_music'
train$Category[train$Category == 'Music/Movie/Game'] <- 'coll_music'
train$Category[train$Category == 'Computer'] <- 'comp_home_sport'
train$Category[train$Category == 'Home/Garden'] <- 'comp_home_sport'
train$Category[train$Category == 'SportingGoods'] <- 'comp_home_sport'
train$Category[train$Category == 'EverythingElse'] <- 'everythingelse'
train$Category[train$Category == 'Health/Beauty'] <- 'health'
train$Category[train$Category == 'Photography'] <- 'photo'

#table(train$Category)

train$endDay[train$endDay == 'Fri'] <- 'Fri_Sun_Tue'
train$endDay[train$endDay == 'Sun'] <- 'Fri_Sun_Tue'
train$endDay[train$endDay == 'Tue'] <- 'Fri_Sun_Tue'
train$endDay[train$endDay == 'Mon'] <- 'Mon_Thu'
train$endDay[train$endDay == 'Thu'] <- 'Mon_Thu'
train$endDay[train$endDay == 'Sat'] <- 'Sat_Wed'
train$endDay[train$endDay == 'Wed'] <- 'Sat_Wed'


#table(train$endDay)

train$Category <- factor(train$Category)
train$endDay <- factor(train$endDay)
train$currency <- factor(train$currency)

fit.full <- glm(Competitive. ~ Category + currency + sellerRating + Duration + endDay + ClosePrice + OpenPrice, data = train, family = binomial(link="logit"))
summary(fit.full)

#########################################  Q1.  ############################################
fit.single <- glm(Competitive. ~ Category, data = train, family = binomial(link="logit"))
summary(fit.single)
#############################################################################################

#########################################  Q2.  ############################################
fit.all <- glm(Competitive. ~ Category + currency + endDay + OpenPrice, data = train, family = binomial(link="logit"))
summary(fit.all)
#############################################################################################

#########################################  Q4.  ############################################
fit.reduced <- glm(Competitive. ~ Category + currency + sellerRating + endDay + ClosePrice + OpenPrice , data = train, family = binomial(link="logit"))
summary(fit.reduced)
anova(fit.reduced, fit.full, test = "Chisq")
#############################################################################################


