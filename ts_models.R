## Cody Crunkilton
## 3/3/20
## PredictIt Notes

library(rtweet)
library(prophet)
library(tidyverse)
library(lubridate)
library(forecast)
library(glue)
# notes -------------------------------------------------------------------
#### markets to look at: 
# 538 approval rating (historical from github)
# tweets

#### models: 
# prophet + ETS/ARIMA
# quasipoisson for overdispersion (tweets build on each other), 
# chance to try out something like xgboost? 

#### model notes:
# for trump, add features for primary election dates, standard holidays

#### notes
# timeframes: daily + weekly - trick is converting everything to noon timeframes
# same model for everyone
# convert to probability distributions: see how those align with the real odds of predictit. 



# data collection: approval ratings --------------------------------------------------------

toplines <- read_csv("https://projects.fivethirtyeight.com/trump-approval-data/approval_topline.csv")

## making the df
prophet_df <- toplines %>% 
  filter(subgroup == 'All polls') %>% 
  mutate(y = approve_estimate,
         ds = mdy(modeldate)) %>% 
  select(ds, y)

#trump$statuses_count # check this matches the basic rules for their page
# model function ----------------------------------------------------------

approval_forecast <- function(prophet_df) {
  
  ### PROPHET SECTIN
  m <- prophet(prophet_df)
  future <- make_future_dataframe(m, periods = 7)
  forecast <- predict(m, future)
  
  # looking at fit and results
  prophet_preds <- forecast %>% 
    as_tibble() %>% 
    left_join(prophet_df %>% mutate(ds = ds %>% as_datetime()), by = 'ds') %>% 
    arrange(ds %>% desc) %>% select(ds, yhat, y, yhat_lower, yhat_upper) 
  
  
  ### ARIMA SECTION
  
  n_ahead = 7
  
  fit <- auto.arima(prophet_df$y)
  #preds <- predict(fit, n.ahead = 7)
  preds <- forecast(fit, h = 7)
  actuals <- c(rep(NA, n_ahead), prophet_df$y)
  
  preds_with_future <- c(fit$fitted, preds$mean)
  
  ### COMBO SECTION
  
  step = prophet_preds %>% filter(is.na(y) == F)
  
  errors <- step$yhat - step$y
  
  arima_errors <- auto.arima(errors)
  
  ae_fit <- predict(arima_errors, n.ahead = n_ahead)
  
  ae_preds <- c(arima_errors$fitted, ae_fit$pred)
  
  ## RESULTS SECTION
  
  results_table <- prophet_preds %>% 
    as_tibble() %>% 
    arrange(ds %>% desc) %>% 
    select(ds, prophet_preds = yhat) %>% 
    mutate(y = actuals,
           arima_preds = preds_with_future,
           #ets_preds = ets_preds,
           combo_preds = prophet_preds - ae_preds,
           lag_preds = y[ds == max(ds[y %>% is.na == F])]) %>% 
    filter(row_number() <= 7)
  
  print('done')
          
  return(results_table)
  
}

# prophet -----------------------------------------------------------------

# fitting and predicting the future
m <- prophet(prophet_df)
future <- make_future_dataframe(m, periods = 7)
forecast <- predict(m, future)

# looking at fit and results
prophet_preds <- forecast %>% 
  as_tibble() %>% 
  left_join(prophet_df %>% mutate(ds = ds %>% as_datetime()), by = 'ds') %>% 
  arrange(ds %>% desc) %>% select(ds, yhat, y, yhat_lower, yhat_upper) %>% print(n = 30)


plot(m, forecast)

prophet_plot_components(m, forecast)


# arima -------------------------------------------------------------------

n_ahead = 7

fit <- auto.arima(prophet_df$y)
preds <- predict(fit, n.ahead = 7)
actuals <- c(rep(NA, n_ahead), prophet_df$y)

preds_with_future <- c(fit$fitted, preds$pred)



# arima of errors ---------------------------------------------------------


step = prophet_preds %>% filter(is.na(y) == F)

errors <- step$yhat - step$y

arima_errors <- auto.arima(errors)

ae_fit <- predict(arima_errors, n.ahead = n_ahead)

ae_preds <- c(arima_errors$fitted, ae_fit$pred)


arima_errors

# ETS ---------------------------------------------------------------------

## this is predicting a +3% jump in approval - not smart, will not include. Also, I know nothing about ETS, so probably shouldn't use it anyway. 

ets_model <- ets(prophet_df$y)

forecast(ets_model, h = 7)

plot(forecast(ets_model, h = 7))

ets_fit <- ets_model$fitted

ets_future <- predict(ets_model, h = 100)

tibble(actuals = ets_future$x, preds = ets_future$fitted)
## note that there are also error bounds here!

ets_preds <- c(ets_future, ets_fit)

length(ets_preds)

length(preds_with_future)

length(ets_fit)


# putting all together ----------------------------------------------------

results_table <- prophet_preds %>% 
  as_tibble() %>% 
  arrange(ds %>% desc) %>% 
  select(ds, prophet_preds = yhat) %>% 
  mutate(y = actuals,
         arima_preds = preds_with_future,
         #ets_preds = ets_preds,
         combo_preds = prophet_preds - ae_preds,
         lag_1 = lead(y),
         lag_7 = lead(y, 7),
         p_errors = abs(prophet_preds - y),
         a_errors = abs(arima_preds - y),
         c_errors = abs(combo_preds - y),
         l1_errors = abs(lag_1 - y),
         l7_errors = abs(lag_7 - y))

results_table

### the combo model has the lowest average/mean/median absolute error before forecasts. The simple lag 1 model does best overall, but combo does better than lag 7
results_table %>% 
  filter(is.na(y) == F &
           row_number() < max(row_number()) - 7) %>% 
  summarise_at(c(#'p_errors', 
                 #'a_errors', 
                 'c_errors', 
                 'l1_errors', 
                 'l7_errors'), c(sum = sum, m = mean, med = median))

## now, need to test a week out for everything at random points


# cross validation --------------------------------------------------------
## to do here: use this to get average errors for the prophet, arima, and simple models at various timeframes. Use that to decide 1) which is best, and 2) find an estimate for the uncertainty. After that, see which is best, and decide whether or not the market behaves rationally!

starts <- seq(ymd('20170601'), ymd('20200310'), by = '1 days')

sampler <- tibble(starts = starts,
                  ends = starts %m+% days(7))

to_test <- sampler %>% sample_n(400)

build_test_df <- function(date) {
  prophet_df %>% 
    filter(ds <= date)
}

enhance_preds <- function(df) {
  df %>% 
    inner_join(prophet_df %>% 
                 mutate(ds = as_datetime(ds)) %>% 
                 select(ds, actual = y)) %>% 
    mutate(y = coalesce(y, actual)) %>% 
    arrange(ds) %>% 
    select(-actual) %>% 
    mutate(arima_error = arima_preds - y,
           prophet_error = prophet_preds - y,
           combo_error = combo_preds - y,
           lag_error = lag_preds - y,
           sample = min(ds),
           rn = row_number()) 
}

## testing every date- save this one cody

tt <- starts %>% #to_test$starts %>% 
  map(build_test_df) %>% 
  map(approval_forecast) %>% 
  map(enhance_preds)

tt %>% bind_rows() %>%  write_csv('/Users/codycrunkilton/Dropbox/personal_github/predictit/data/approval_ratings_testing.csv')

tt %>% 
  bind_rows() %>% 
  group_by(rn) %>% 
  summarise_at(c('arima_error', 'prophet_error', 'combo_error', 'lag_error'), function(x) {mean(abs(x))})

## looks like taking the value at n-1 is more accurate than anything else!
## and, ARIMA beats prophet and combo. 

## is there any variance over time? Doesn't look like it
tt %>% 
  bind_rows() %>% 
  mutate(month = floor_date(sample, 'months')) %>% 
  group_by(month) %>% 
  summarise_at(c('arima_error', 'prophet_error', 'combo_error', 'lag_error'), function(x) {mean(abs(x))}) %>% 
  pivot_longer(names_to = 'error', values_to = 'values', -c(month)) %>% 
  ggplot(aes(x = month, y = values)) + 
  geom_col() +
  facet_wrap(~error)

## win percent:
tt %>% 
  bind_rows() %>% 
  select(ds, sample, rn, arima_error:lag_error) %>% 
  pivot_longer(names_to = 'error', values_to = 'values', -c(ds, sample, rn)) %>% 
  group_by(sample, ds) %>% 
  arrange(abs(values)) %>% 
  filter(row_number() == 1) %>% 
  group_by(error) %>% 
  summarise(n_wins = n()) %>% 
  arrange(n_wins) %>% 
  mutate(win_pct = 100 * n_wins/sum(n_wins))

## looks like the simple lag model is the best almost 50% of the time - the others are fairly evenly split, but arima > combo > prophet

## making a graph:


# The simple model, which won: ----------------------------------------------------------

lag_model <- prophet_df %>% 
  mutate(lag1 = lead(y, 1),
         lag2 = lead(y, 2),
         lag3 = lead(y, 3),
         lag4 = lead(y, 4),
         lag5 = lead(y, 5),
         lag6 = lead(y, 6),
         lag7 = lead(y, 7),
         #blend = ma(lead(y, 10), 7 %>% as.numeric()
         ) %>% 
  pivot_longer(names_to = 'lag', values_to = 'pred', -c(ds, y)) %>% 
  mutate(error = pred - y)

lag_model %>% 
  group_by(lag) %>% 
  summarise(error = mean(abs(error), na.rm = T))

lag_model %>% 
  ggplot(aes(x = error)) + 
  geom_histogram() + 
  facet_wrap(~lag)

## blending doesn't matter, and the further back you go the higher the error is. Let's look at some plots:

lag_model %>% 
  mutate(month = floor_date(ds, 'months')) %>% 
  group_by(month, lag) %>% 
  summarise(avg_error = mean(abs(error))) %>% 
  ggplot(aes(x = month, y = avg_error)) + 
  geom_col() + 
  facet_wrap(~lag)

## looks like the variance has decreased slightly since it started - close enought to being even though. might be worth excluding the first month or two. 


# for predictions ---------------------------------------------------------

lag_limited <- lag_model %>% filter(ds > min(ds) %m+% months(2))

lag_small <- lag_limited %>% 
  group_by(lag) %>% 
  summarise(sd = sd(error, na.rm = T))

lag_all <- lag_model %>% 
  group_by(lag) %>% 
  summarise(sd = sd(error, na.rm = T))

lag_all

lag_model$error %>% sd(na.rm = T)

lag_model$error %>% mean(na.rm = T)

## predicting 3/5:
lag_small
lag_all

new_dates <- seq(max(prophet_df$ds) %m+% days(1), max(prophet_df$ds) %m+% days(7), by = '1 days')

tibble(ds = new_dates, 
       y = prophet_df$y[prophet_df$ds == max(prophet_df$ds)],
       small_sd = lag_small$sd, 
       sd = lag_all$sd)

## to do: get function that takes target date, predictit cutoffs, and estimates the probabilities for each under both the big_sd and small_sd segments. Or, do that in python and have this end here. (Or, move everything to python- this is all simple)

## get cutpoints predictit uses, predict those values

## pretend that cutpoints are by .3 increments from 41.5 to 45.5
## (pull this from predictit API then do this step):
cutpoints <- seq(43.4, 45.8, by = .4) # from lowest to second highest

## remember they will round, fix that later
prophet_df
sd_dev = .232
current_price = prophet_df$y[1]
current_price

simple_model <- tibble(lower = cutpoints, 
       upper = cutpoints + .4,
  ) %>% 
  bind_rows(tibble(lower = c(0, max(cutpoints)), upper = c(min(cutpoints), 100)) )%>% 
  mutate(lw = lower - .05,
         up = upper - .05,
       prob = ((pnorm(up-current_price, sd = sd_dev, lower.tail = T) - pnorm(lw-current_price, sd = sd_dev, lower.tail = T)) * 100) %>% round(1),
       real_upper = upper - .1 # doing this to make them match predictit's format
       ) %>% 
  arrange(lower, upper) %>% 
  select(lower, upper = real_upper, prob)

simple_model

simple_model %>% 
  mutate(predicit_probs)

pnorm(45.8, mean = 45.5, sd = .269, lower.tail = T)

# unsorted ----------------------------------------------------------------

gen_arima_preds <- function(t_acq, fit) {
  
  newxreg_arima <- t_acq %>% filter(y %>% is.na == T) %>% ungroup %>% select(cost_scaled, quantity_scaled)
  n_ahead <- nrow(newxreg_arima)
  non_na <- t_acq$y[t_acq$y %>% is.na == F] %>% length
  
  if(n_ahead > 0) {
    arima_future <- predict(fit, n.ahead = n_ahead, newxreg = newxreg_arima)
  } else {
    arima_future <- NULL
  }
  
  arima_preds <- fit$fitted[fit$fitted %>% is.na == F]
  final_arima <- c(arima_preds, arima_future$pred)
  
  return(final_arima)
  
}

prophet_errors <- forecast$yhat - df$y
error_arima <- auto.arima(prophet_errors[prophet_errors %>% is.na == F])
error_arima_past <- error_arima$fitted
error_arima_future <-  predict(error_arima, n.ahead = length(prophet_errors[prophet_errors %>% is.na == T]))

arima_resids <- c(error_arima_past[error_arima_past %>% is.na == F], error_arima_future$pred)



# notes: Approval Rating --------------------------------------------------

## convert to probability distributions
## estimate cross-validated error
## try adding on a post-hoc ARIMA
## estimate the error a second time to see if it's better
## add holidays, check if that helps

#### check vs: 
## only arima
## only ETS

#### think about:
## incorporating poll release info before they update it
## add another penalty to account for peopel who know insider info about polls that come out

#### vs predictit:
## compare to historical prices (might not be possible)
## save these predictions as csv/whatever, grab predictit prices, see if there are divergences
