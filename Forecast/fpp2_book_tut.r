library('fpp2')

library('ggplot2')
library('forecast')
library('tseries')
library('dplyr')
library('doParallel')
library('tidyverse')
library('magrittr')

y <-  ts(rnorm(13), start= 2018, end = 2020, frequency  = 52)
ts.plot(y)


autoplot(melsyd[,"Economy.Class"]) +
  ggtitle("Economy class passengers: Melbourne-Sydney") +
  xlab("Year") +
  ylab("Thousands")

autoplot(a10) +
  ggtitle("Antidiabetic drug sales") +
  ylab("$ million") +
  xlab("Year")

ggseasonplot(a10, year.labels=TRUE, year.labels.left=TRUE) +
  ylab("$ million") +
  ggtitle("Seasonal plot: antidiabetic drug sales")


createDF <- function()
{
  rst <- '/home/emre/fMRI/Codes_GH/Getron/data/rst.rpt'
  DF <- readr::read_table2(rst, skip = 6, col_names = FALSE)
  colndf <-
    readr::read_table2(rst,
                       skip = 4,
                       n_max = 10,
                       col_names = TRUE)
  colnames(DF) <- colnames(colndf)
  
  DF <-  DF %>% subset(!is.na(DF$BDIntCode))
  # DF <-
  #   DF %>% mutate(NC =  paste(BDIntCode, ForecastPcal, sep = "-"))
  # DF$NC <- as.factor(DF$NC)
  
  
  # DF$BDIntCode <- NULL
  # DF$ForecastPcal <- NULL
  
  # DF <- DF %>%
  #   group_by(NC) %>%
  #   arrange(PCalOrder, .by_group = TRUE)
  # 
  # lvl <-  levels(DF$NC)
  DF <-  ungroup(DF)
  DF
}

DF <- createDF()
DF$ForecastPcal <-  NULL
DF.uniq <- unique(DF)



#DF2 <- DF.uniq %>% group_by(BDIntCode) %>% summarize(MN = mean(BDPropertyValue)) %>% filter(MN > 50)
DF2 <- DF.uniq %>% group_by(BDIntCode) %>% summarise(SM = sum(BDPropertyValue)) %>% arrange(desc(SM)) %>% filter(SM > 50) %>% ungroup()
DF3 <- DF.uniq %>% filter(BDIntCode %in% DF2$BDIntCode)
DF3 %<>% mutate(sclaled.sales = scale(BDPropertyValue))

DF %>%  filter(BDIntCode == DF2$BDIntCode[120]) %>% ggplot(aes(x = PCalDIntCode, y = BDPropertyValue)) + geom_line()

A <- read.csv2('../data/SALE15.csv',sep = ',',encoding = 'UTF-8-BOM')
A %<>% rename(RelatedBDIntCode = ï..RelatedBDIntCode)

A %>% group_by(RelatedBDIntCode) %>% count() %>% View()

A1 <- A %>% filter(RelatedBDIntCode=='52654')
A2 <- A %>% filter(RelatedBDIntCode=='36409')


A1.ts <- ts(A1$SALE, 
   freq = 52, 
   start = c(2016,1),
   end = c(2020,1))

ggseasonplot(A1.ts, year.labels=TRUE, year.labels.left=TRUE) +
  ylab("Unit") +
  ggtitle("Seasonal plot: Don")

xts

A2.ts <- ts(A2$SALE, 
            freq = 52, 
            start = c(2016,1),
            end = c(2020,1))

ggseasonplot(A2.ts, year.labels=TRUE, year.labels.left=TRUE) +
  ylab("Unit") +
  ggtitle("Seasonal plot: Gomlek")

ggseasonplot(A2.ts, year.labels=TRUE, year.labels.left=TRUE,polar = T) +
  ylab("Unit") +
  ggtitle("Polar Seasonal plot: Gomlek")

ggsubseriesplot(A2.ts) +
  ylab("Unit") +
  ggtitle("Seasonal subseries plot: Gomlek")

A1.ts

autoplot(elecdemand[,c("Demand","Temperature")], facets=TRUE) +
  xlab("Year: 2014") + ylab("") +
  ggtitle("Half-hourly electricity demand: Victoria, Australia")

autoplot(A1.ts, facets=TRUE) +
  xlab("Tum Yillar") + ylab("") +
  ggtitle("Don: Yillik")

gglagplot(A1.ts)

ggAcf(A1.ts)
acf(A1.ts)


naive(A1.ts, 5)
rwf(A1.ts, 5) # Equivalent alternative
snaive(A1.ts, 5)

rwf(A1.ts, 5, drift=TRUE)

# Set training data from 1992 to 2007
beer2 <- window(ausbeer,start=1992,end=c(2007,4))
# Plot some forecasts
autoplot(beer2) +
  autolayer(meanf(beer2, h=11),
            series="Mean", PI=FALSE) +
  autolayer(naive(beer2, h=11),
            series="Naïve", PI=FALSE) +
  autolayer(snaive(beer2, h=11),
            series="Seasonal naïve", PI=FALSE) +
  ggtitle("Forecasts for quarterly beer production") +
  xlab("Year") + ylab("Megalitres") +
  guides(colour=guide_legend(title="Forecast"))

# Set training data from 1992 to 2007

# Plot some forecasts
autoplot(A1.ts) +
  autolayer(meanf(A1.ts, h=20),
            series="Mean", PI=FALSE) +
  autolayer(naive(A1.ts, h=20),
            series="Naïve", PI=FALSE) +
  autolayer(snaive(A1.ts, h=20),
            series="Seasonal naïve", PI=FALSE) +
  ggtitle("Forecasts for weekly sales") +
  xlab("Weeks") + ylab("Units") +
  guides(colour=guide_legend(title="Forecast"))

library(gtrendsR)
search_terms <- "Bipolar Disorder"

output_results <- gtrends(keyword = search_terms,
        geo = "TR",
        time = "all") -> output_results

output_results %>%
  .$interest_over_time %>%
  ggplot(aes(x = date, y = hits)) +
  geom_line(colour = "darkblue", size = 1.5) +
  facet_wrap(~keyword) +
  ggthemes::theme_economist() -> plot

plot

forecast(auto.arima(output_results$interest_over_time$hits),5)

library(gtrendsR)

search_terms <- "dildo"
output_results <- gtrends(keyword = search_terms,geo = "US",
                          time = "all")  


hitDF <-
  data.frame(
    hits = output_results$interest_over_time$hits,
    date = as.Date(output_results$interest_over_time$date),
    year = lubridate::year(as.Date(output_results$interest_over_time$date))
  )
hitDF2010 <- hitDF %>% filter(year > 2010)

hitDF2010 %>% ggplot(aes(x = date, y = hits)) +
  geom_line(colour = "darkblue", size = 1.5) +
  #facet_wrap(~keyword) +
  ggthemes::theme_economist() -> plot
plot

hitDF2010 %>% glimpse()


BP.ts <- ts(hitDF2010$hits, 
            freq = 12, 
            start = c(2011,1))

ggseasonplot(BP.ts, year.labels=TRUE, year.labels.left=TRUE) 

ggsubseriesplot(BP.ts) 


