---
title: "Tweet Sentiment Analysis"
layout: single
classes: wide
author_profile: true


header:
  teaser: /assets/img/tweets/Twitter_Logo_Blue.png
---

The sentiment analysis of Tweet posts was an individual assignment completed for my Principles of Data Science class. This "project" was broken into three parts:

1. Downloading and Creating the Dataframe: 

Because the Trump tweets data were downloaded as a json table, I had to reconstruct the dataframe using Pandas. The index of this dataframe was the ID labels of each tweet, and contains the following columns: 

	- time: The time the tweet was created encoded as a datetime object.
	- source: The source device of the tweet.
	- text: The text of the tweet.
	- retweet_count: The retweet count of the tweet.

In the process of cleaning up the provided data, I had to extract the source device of the tweets using regex patterns: 

For instance, "trump['source'].unique()" results in an array containing the strings of where each tweet source came from (as links) (e.g. "<a(space)href="http://twitter.com/download/android" rel="nofollow">Twitter for Android</a>"). Notice that the "Twitter for Android" (aka the source) is surrounded by HTML tags. These phrases were extracted using the specified regex pattern:  "\<[^>]*[^>]\>". The resulting dataframe "trump" is shown below. 


<img src="/assets/img/tweets/trump_df.png">


2. Tweet Source Analysis (visualizations)

The goal is to determine if there was a difference between Trump's tweet behavior across different devices. 


First we see that there are sources that are used much more than others from the following plot. 

<img src="/assets/img/tweets/tweet_source_bar.png">


A sns.displot also shows the change in the distribution of tweet sources over the years. 

<img src="/assets/img/tweets/dist_source_all_years.png">


Taking the two main sources used to post tweets, we also show the distribution over hours of the day (in eastern time) for each device. 

<img src="/assets/img/tweets/line_dist_all_source.png">

Now, according to a verge article (link), that states that Trump may have switched over from an Android to an iPhone sometime in 2017. We have created a similar distribution as shown before, but using tweets before 2017. 

<img src="/assets/img/tweets/pre_seventeen.png">


It was also theorized that during his campaign, Trump's tweets were written by different people. Those from Android sources were written by him personally, and those from iPhones may have been written from his staff. Of course, there is no way of proving this directly, but we may be able to infer from our visualizations that different people were responsible for Trump's tweet posts. In other words, there are overlaps in the hour where multiple texts from different devices were sent/tweeted out at the same time (see figures above). We could also possibly plot a scatter plot for minutes, and look specifically at any overlap points. That is, if a scatter plot of points show overlap on messages sent for both androids and iphones in the same minute, this could support the claim that tweets were sent be different people. Perhaps this could be conducted in a future analysis? 

3. Sentiment Analysis 

In the third and final part of this exercise, we calculated the sentiment of Trump's tweets by assigning a sentiment to each of word using [VADER (Valence Aware Dictionary and Sentiment Resoner)](https://github.com/cjhutto/vaderSentiment). Note that the format of the vader_lexicon.txt file is organized as follows: A word (including emojis), followed by its mean sentiment rating, standard deviation, and raw human-sentiment-rating. You can read more about it in the link.

After removing the punctuation from the text/tweet posts (to match words properly), I converted the tweets into a [tidy format](https://cran.r-project.org/web/packages/tidyr/vignettes/tidy-data.html) so that it would be easier to match words. 

Fig. 1: Tidy Data
<center><img src="/assets/img/tweets/tidy_format.png"></center>


Fig. 2: Tweets with Polarity
<img src="/assets/img/tweets/polarity_df.png" align="left">


With this polarity dataframe, we're able to determine the most negative and most positive tweets: 

Most negative tweets:

	   the trump portrait of an unsustainable border crisis is dead on. “in the last two
	   years, ice officers made 266,000 arrests of aliens with criminal records, including 
	   those charged or convicted of 100,000 assaults, 30,000 sex crimes &amp; 4000 violent 
	   killings.” america’s southern....

	   it is outrageous that poisonous synthetic heroin fentanyl comes pouring into the u.s. 
	   postal system from china. we can, and must, end this now! the senate should pass the 
	   stop act – and firmly stop this poison from killing our children and destroying our 
	   country. no more delay!

	   the rigged russian witch hunt goes on and on as the “originators and founders” of this 
	   scam continue to be fired and demoted for their corrupt and illegal activity. all 
	   credibility is gone from this terrible hoax, and much more will be lost as it 
	   proceeds. no collusion!

	   ...this evil anti-semitic attack is an assault on humanity. it will take all of us 
	   working together to extract the poison of anti-semitism from our world. we must unite 
	   to conquer hate.

	   james comey is a proven leaker &amp; liar. virtually everyone in washington thought 
	   he should be fired for the terrible job he did-until he was, in fact, fired. he 
	   Leaked classified information, for which he should be prosecuted. he lied to congress 
	   under oath. he is a weak and.....


Most positive tweets:

	   congratulations to patrick reed on his great and courageous masters win! when patrick 
	   had his amazing win at doral 5 years ago, people saw his great talent, and a bright 
	   future ahead. now he is the masters champion!

	   congratulations to a truly great football team, the clemson tigers, on an incredible 
	   win last night against a powerful alabama team. a big win also for the great state of 
	   south carolina. look forward to seeing the team, and their brilliant coach, for the 
	   second time at the w.h.

	   my supporters are the smartest, strongest, most hard working and most loyal that we 
	   have seen in our countries history. it is a beautiful thing to watch as we win 
	   elections and gather support from all over the country. as we get stronger, so does 
	   our country. best numbers ever!

	   thank you to all of my great supporters, really big progress being made. other 
	   countries wanting to fix crazy trade deals. economy is roaring. supreme court pick 
	   getting great reviews. new poll says trump, at over 90%, is the most popular 
	   republican in history of the party. wow!

	   thank you, @wvgovernor jim justice, for that warm introduction. tonight, it was my 
	   great honor to attend the “greenbrier classic – salute to service dinner” in west 
	   virginia! god bless our veterans. god bless america - and happy independence day to 
	   all! https://t.co/v35qvcn8m6





**Citations**: 
- Rights/credits to the creation of this assignment goes to UC Berkeley's DS100 class (Spring 2019) (link here)
- Tweet logo taken from Twitter