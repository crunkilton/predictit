## Trying to figure out how to recover tweets older than the most recent 3200
## Tried two things here, neither really works
## Probably too much work, will revisit later
## this script is a mess, don't use it for much

## GOT doesn't get everything

## this gets stuff, but doesn't get retweets. Why no retweets? how can I find the element for retweets?

## test pulling retweet times from here - probably not worth the effort though. can get all of Trumps for a test. 

## look at GOT again. 

from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.common.exceptions import NoSuchElementException, StaleElementReferenceException
from time import sleep
import json
import datetime


# edit these three variables
user = 'AOC'
start = datetime.datetime(2020, 3, 8)  # year, month, day
end = datetime.datetime(2020, 3, 10)  # year, month, day

# only edit these if you're having problems
delay = 5  # time to wait on each page load before reading the page
driver = webdriver.Safari()  # options are Chrome() Firefox() Safari()


# don't mess with this stuff
twitter_ids_filename = 'all_ids.json'
days = (end - start).days + 1
tweet_selector = 'article > div > div > div > div > div > div > a'
user = user.lower()
ids = []

def format_day(date):
	day = '0' + str(date.day) if len(str(date.day)) == 1 else str(date.day)
	month = '0' + str(date.month) if len(str(date.month)) == 1 else str(date.month)
	year = str(date.year)
	return '-'.join([year, month, day])

def form_url(since, until):
	#p1 = 'https://twitter.com/search?f=tweets&vertical=default&q=from%3A'
	p1 = 'https://twitter.com/search?q=from%3A'
	p2 =  user + '%20since%3A' + since + '%20until%3A' + until + '%20include%3Anativeretweets&src=typd'
	return p1 + p2

def increment_day(date, i):
	return date + datetime.timedelta(days=i)

for day in range(days):
	d1 = format_day(increment_day(start, 0))
	d2 = format_day(increment_day(start, 1))
	url = form_url(d1, d2)
	print(url)
	print(d1)
	driver.get(url)
	sleep(delay)
	try:
		found_tweets = driver.find_elements_by_css_selector(tweet_selector)
		all_tweets = found_tweets[:]
		increment = 0
		
		while len(found_tweets) >= increment:
			print('scrolling down to load more tweets')
			driver.execute_script('window.scrollTo(0, document.body.scrollHeight);')
			sleep(delay)
			found_tweets = driver.find_elements_by_css_selector(tweet_selector)
			all_tweets += found_tweets[:]
		
			print('{} tweets found, {} total'.format(len(found_tweets), len(ids)))
			increment += 10   
		
		for tweet in all_tweets:
			try:
				id = tweet.get_attribute('href').split('/')[-1]
				ids.append(id)
			except StaleElementReferenceException as e:
				print('lost element reference', tweet)
		
		print(ids)

	except NoSuchElementException:
		print('no tweets on this day')
	start = increment_day(start, 1)

finalids = np.unique([tweetid for tweetid in ids if tweetid.isdigit() == True]).tolist()

try:
    with open(twitter_ids_filename) as f:
        all_ids = finalids + json.load(f)
        data_to_write = list(set(all_ids))
        print('tweets found on this scrape: ', len(finalids))
        print('total tweet count: ', len(data_to_write))
except FileNotFoundError:
    with open(twitter_ids_filename, 'w') as f:
        all_ids = finalids
        data_to_write = list(set(all_ids))
        print('tweets found on this scrape: ', len(finalids))
        print('total tweet count: ', len(data_to_write))

with open(twitter_ids_filename, 'w') as outfile:
    json.dump(data_to_write, outfile)

print('all done here')
driver.close()
