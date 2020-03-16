## Cody Crunkilton
## 3/6/20
## Data collection: Tweet scraping historical tweets for everyone in predictit. 

# This is the scrape of everyone - 

library(rtweet)
library(prophet)
library(tidyverse)
library(lubridate)
library(forecast)
library(glue)


# below is getting everything ---------------------------------------------

## set an update method to get this

for (user in users) {
  
  test = get_timeline(user, 1)
  
  max_id = test$status_id
  
  max_tweets = lookup_users(user)$statuses_count
  
  iters = ceiling(max_tweets/3200)
  
  all_tweets = tibble()
  
  for (i in 1:1){
    print(glue('getting {user} tweets, wave {i} of {iters} at {Sys.time()}'))
    x = get_timeline(user, n = 3800, max_id = max_id)
    
    max_id = min(x$status_id)
    
    all_tweets = all_tweets %>% 
      bind_rows(x)
    
    all_tweets%>% select(user_id:retweet_count) %>% write_csv(glue('/Users/codycrunkilton/Desktop/holder/tweets/{user}_tweets.csv'))
    
    print(glue('sleeping 15 minutes at {Sys.time()}'))
    Sys.sleep(1 * 60)
  }
}

all_tweets %>% select(user_id:retweet_count) %>% write_csv('/Users/codycrunkilton/Desktop/holder/test.csv')

test <- read_csv('/Users/codycrunkilton/Desktop/holder/tweets/aoc_tweets.csv')
test

all_tweets %>% 
  select_if(is.list(col) == F)
filter(class == list())


colnames(all_tweets)

x$created_at %>% min

x = get_timeline('aoc', n = 10, max_id = )

x$status_id

## Pence: 8k tweets
## 

# misc notes to clean up later  ------------------------------------------------

## this didn't get everything, clean it up later. 

#GetOldTweets3 --username "aoc,berniesanders,donaldtrumpjr,whitehouse,joebiden,mike_pence,potus,realdonaldtrump"  --maxtweets 10000000000000 --output all_tweets.csv

#GetOldTweets3 --username "aoc"  --maxtweets 10000000000000 --output aoc_tweets.csv

## this misses tweets: says potus only has 424 tweets. 

## use https://github.com/bpb27/twitter_scraping


## this will take a while - save all tweets is time consuming.
## for trump: use trump twitter archive: https://github.com/mkearney/trumptweets
get_timeline('donaldtrumpjr')

# pulling the tweets ------------------------------------------------------

## AOC: 10k tweets
users <- c(#'aoc', 'berniesanders', 
           'donaldjtrumpjr', 
           'whitehouse', 'joebiden', 'mike_pence', 'potus') # because AOC was the test. I suspect 'mikebloomberg' will drop off later so not bothering with that either. realDonaldTrump is accessible from trump twitter archive

aoc2 %>% arrange(created_at %>% desc) %>% select(created_at, text)

aoc %>% arrange(date %>% desc) %>% select(date, text)

aoc %>% 
  mutate(status_id = id %>% as.character()) %>% 
  anti_join(aoc2, by = c('date' = 'created_at')) %>% 
  select(text, date, id) %>% 
  arrange(date %>% desc)

aoc2 %>% filter(text %>% str_detect('HolocaustMemorialDay')== T)

max_id = 1235590684935143424

get_timeline('aoc', n=10, max_id = 123559068493514342)

lookup_tweets('aoc')


# other pulling testing ---------------------------------------------------
## didn't work

test1 <- lookup_statuses(c('1235780866409926657',
                          '1235755461154148352',
                          '1236063858407718914',
                          '1236061629269319680',
                          '1236049597971259395',
                          '1235780866409926657',
                          '1235755461154148352',
                          '1236063858407718914',
                          '1236061629269319680',
                          '1236049597971259395',
                          '1236101614181720069',
                          '1236148947607056385',
                          '1236134458471915520',
                          '1236102826952200193',
                          '1236101614181720069',
                          '1236148947607056385',
                          '1236134458471915520',
                          '1236102826952200193',
                          '1236533635290890240',
                          '1236462881828454401',
                          '1236533635290890240',
                          '1236462881828454401'))

test1 %>% select(text)
test1 %>% select(status_id, screen_name, created_at, text) %>% arrange(created_at %>% desc)

test1 <- lookup_statuses(c('1236533635290890240',
                           '1236533635290890240',
                           '1236745013192732679',
                           '1236859141223743490',
                           '1236745013192732679',
                           '1236859141223743490'))

test1 %>% select(status_id, created_at, text, retweet_status_id:retweet_description)

aoc <- get_timeline('aoc')

aoc %>% select(status_id, is_retweet, created_at)
