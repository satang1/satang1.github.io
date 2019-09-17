---
title: "Spam & Ham Classification"
layout: single
classes: wide
author_profile: true


header:
  teaser: /assets/img/spam_ham/email_logo.png
---

The aim of this spam/ham classification project was to create a classifier that would distinguish between spam (junk mail) and ham mail. By the end of this project, I was comfortable with (1) feature engineering with text data (2) Using sklearn libraries to process data and fit models (3) Validating the performance of my model and minimizing overfitting (4) Generating and analyzing precision-recall curves. 

**Part 1) Initial Analysis and Basic EDA**


The dataset I used consisted of 8348 (training set) labeled examples and 1000 unlabeled emails (test set). The dataframe created contained 4 initial columns: id (identifier of training example), subject (subject of email), email (text of the email), spam (1 if the email is spam, and 0 if ham). Note that the results of my predictions were submitted to Kaggle for evaluation, where I received over 90% accuracy in classifying spam/ham emails. A summary of my process, thoughts, and evaluations are discussed below. 

<img src="/assets/img/spam_ham/initial_df.png">

As a preliminary check, I considered whether there were any missing or NaN values in the dataframe, and filled in any necessary information. (i.e. 6 rows in the "subject" column were filled in with .fillna). 

And if I take a look at the first ham and first spam email texts of the training set, I am able to infer some potential patterns which may help to differentiate between ham/spam emails: 

First Ham Email: 

	url: http://boingboing.net/#85534171
	 date: not supplied
	 
	 arts and letters daily, a wonderful and dense blog, 
	 has folded up its tent due 
	 to the bankruptcy of its parent company. a&l daily 
	 will be auctioned off by the 
	 receivers. link[1] discuss[2] (_thanks, misha!_)
	 
	 [1] http://www.aldaily.com/
	 [2] http://www.quicktopic.com/boing/h/zlfterjnd6jf
	 
	 

First Spam Email:

	<html>
	 <head>
	 </head>
	 <body>
	 <font size=3d"4"><b> a man endowed with a 7-8" 
	 hammer is simply<br>
	  better equipped than a man with a 5-6"hammer. <br>
	 <br>would you rather have<br>more than enough to 
	 get the job done or fall =
	 short. it's totally up<br>to you. our methods are 
	 guaranteed to increase y=
	 our size by 1-3"<br> <a 
	 href=3d"http://209.163.187.47/cgi-bin/index.php?10=
	 004">come in here and see how</a>
	 </body>
	 </html>


Looking at the content of the two emails, the first email (which is not spam) is mainly telling the reader news about what is happening in the world, or relaying pertinent information. Therefore, the message is straight to the point, and less wordy when compared to the spam email which uses "persuasive" words like "guarantee" and "more than enough", etc. The spam email is also noticeably longer and wordier than the ham email.


At this point, I conducted at training validation split with the sklearn.model selection train_test_split method for a test size of 0.1. This is so that I will have a validation dataset to assess the performace of my classifer once I am done training my classifier. 

**Basic Feature Engineering**

Since this is a classic spam/ham classification problem, I will be using logistic regression to train my classifer. To this, I will need a numeric feature matrix X and a vector of corresponding binary labels y. 


To create one of my initial features, I created a function that would take in take in a list of words and a panda Series of email texts. The function would then ouput a 2d array, with one row for each email, and each row would contain a 0 or a 1 (denoting whether the word is in the text or now). 

For example, if the words inputted were ["happy", "sad", "confused"], and the email series were pd.Series(["happy today", "don't be sad"]), then the ouputted array would be as follows: 

	array([1, 0, 0], #for the first email text 
	      [0, 1, 0]) #for the second email text


It's also convenient to display a sns.barplot to compare the proportion of emails which contain certain words. For instance, you can see from the below plot that "free" shows up more in spam emails than in ham emails. 


<img src="/assets/img/spam_ham/prop_bar.png">

I've also created a class conditional density plot like the one shown below which compares the distribution of spam emails to the length of the ham amils in the training set. 


<img src="/assets/img/spam_ham/dist_plot.png">


If I take a look at a boxplot of spam/ham email lengths, it is also obvious that spam emails are wordier than ham emails. # This is because the median (50% line) is much higher than the median line of the ham box plot. Also, excluding the outliers (which are present in both ham and spam), the upper quartile of the boxplot on the spam is much longer than the upper quartile of the ham boxplot, which indicates that 25% of the data resides in this interval and is much larger/wordier than ham emails. This is a good indicator that emails with lengths greater than 3000 are spam emails. (You can also see this disparity by calculating the mean email length, which verifies that spam length is usually wordier than ham length). 


<img src="/assets/img/spam_ham/bar_spam.png">


Some of the other features that I selected (and perhaps converted into numeric values) were the the email message length, the subject message length, if there was a reply to the email, whether there were special characters (i.e. %$/@ signs) in the subject and/or email text. I've also included the digit count (i.e. counted any numbers) and uppercases in the text and subject lines of the emails. In the process of creating this featrure matrix, I also realized that thresholding the values was a very effective way of increasing classification accuracy. 

For instance, if the email message length was greater than, say, 3000 characters in length, then I would ouput a 1 (as spam), and 0 (as ham) otherwise. This gives greater meaning to the value of message lengths when used in the features matrix for classification. The same thresholding concept was also applied to the subject length and the number of uppercase letters in a message line. 

My resulting matrix looks something like this: 

	array([[  0.,   1.,   0., ...,  54.,   1.,   1.],
	       [  1.,   0.,   0., ..., 148.,   1.,   1.],
	       [  0.,   1.,   0., ...,  21.,   0.,   1.],
	       ...,
	       [  0.,   0.,   0., ...,  15.,   0.,   1.],
	       [  0.,   0.,   0., ...,  23.,   0.,   1.],
	       [  0.,   0.,   0., ...,  13.,   0.,   1.]])

**Classification**

In this part of the project, I processed my training set data to create the corresponding feature matrix X. Then I assigned my binary labels, y, to the corresponding "spam" (0 or 1) values from the training set. With this, I fitted my model with a logisitic regression model from sklearn and evaluated the accuracy with my validation set data. In the end, I recieved about a 90% average accuracy for my logistic classifcation. 

#------------TRAINING DATA ON TRAIN SET----------

	features = processing(train)

	x_train = create_X_matrix(features)

	y_train = train["spam"].values

	test_model = LogisticRegression(C=2)
	test_model.fit(x_train, y_train)
	test_model.score(x_train, y_train)
	Output: 0.9178756821509384 



#------------TRAINING DATA ON VALIDATION SET----------

	val_features = processing(val)

	x_train_val = create_X_matrix(val_features)

	y_train_val = val["spam"].values


	test_model.score(x_train_val, y_train_val)
	Output: 0.9065868263473054


I will also point out that I compared my model with a zero_predictor classifer to evaluate the precision, recall and false alarm rate of the email classifications. First I will define the following: 

<li> False positives: ham getting classified as spam (not in inbox) </li> 
<li> False negatives: spam getting classified as ham (in inbox)</li> 


<li> Precision = TP/ (TP + FP) proportion of emails flagged as spma that are actually spam </li>

<li> Recall = TP / (TP + FN) proportion of spam emails that were correctly flagged as spam </li>

<li> False alarm rate = FP / (FP + TN) proportion of ham emails that were incorrectly flagged as spam. </li>


#Calculations (6a)

	zero_pred = np.zeros(len(train["spam"]))
	actual_y = train["spam"]

	TP = np.count_nonzero((actual_y == zero_pred) & (zero_pred == 1))

	TN = np.count_nonzero((actual_y == zero_pred) & (zero_pred == 0))

	FP = np.count_nonzero((actual_y != zero_pred) & (zero_pred == 1))

	FN = np.count_nonzero((actual_y != zero_pred) &(zero_pred == 0)) 


	zero_predictor_fp = FP
	zero_predictor_fn = FN

	print("false_pos: ", zero_predictor_fp, "; false_neg: ", zero_predictor_fn)
	Output: false_pos:  0 ; false_neg:  1918


#More Calculations: (6b)

	#https://www.lexjansen.com/nesug/nesug10/hl/hl07.pdf based on these definitions

	#acc: (TP + TN)/ (TP + TN + FP + FN) 
	#recall: TP/(TP + FN)

	zero_predictor_acc = (TP + TN) / (TP + TN + FP + FN) 
	zero_predictor_recall = TP/ (TP + FN)

	print("accuracy: ", zero_predictor_acc, "; recall: ", zero_predictor_recall)
	Output: accuracy:  0.7447091707706642 ; recall:  0.0


After creating out zero_predictor, I briefly discussed my observations and why I observed what I saw in 6a and 6b: 
<ul>
<li> False positives (FP) are the number of emails that are labeled/classifed/predicted to be spam (i.e. 1 or positive) when they are actually ham (i.e. 0 or negative), and since our zero_predictor is only going to predict 0, I will have no false positives because it will never classify or predict 1. </li>

<li> False negatives (FN) are the number of emails that are labled/predicted to be ham (i.e. 0 or negative) when in reality the email is a spam (i.e 1 or positive). And thus, from the number calculated in 6a, I see that there are 1918 emails which are incorrectly labeled as ham when they are really spam emails. </li>

<li> Accuracy is the proportion of correctly classified emails for both spam and ham (i.e. spam is correctly classified as spam (TP) and ham is correctly classified as ham (TN)), and thus I want the proportion of all the true positives and true negatives over all the emails classified (i.e. (TP + TN)/ (TP + TN + FP + FN)). From our calculation in 6b, I see that I acheive an accuracy rate of about 74% if I use a zero only classifier. </li>

<li> Recall (or the sensitivity) measures how good our zero-predictor classifier is at predicting a spam email (positive rate). It is the proportion of correctly classified positive emails (correct spams) over all the emails classified as spam (i.e. positive). But since our zero-predictor will never predict a 1 (or positive), I don't have any true positives and thus our recall percentage is 0. </li>
</ul>


Therefore, in comparision to a standard zero_predictor classifier, which only recieves about a 74% accuracy, our logisitic regression classifier recieves a 90% accuracy. 


**Summary of Feature/Model Selection Process:** *(i.e. What I learned)*

1. How did I find better features for my model? 
2. What did I try that worked/didn't work?
3. What was surprising in my search for good features? 

<ol>
	<li> To look for better features, I visualized several bar/count plots and distplots using seaborn to see the difference between spam and ham emails for that specific feature. For instance, when I visualized the distribution/proportion of message length in emails for spam and ham, I realized that if the email length was over 3000 words long, then it was mostly likly a spam email (I also compared statistics to determine this too (i.e. mean lengths)). Another thing I did was read through some of the spam emails (including the subject descriptions) to look for any patterns or unusual features (things that stand out), so that I could use the distinction/feature as a way to differentiate between spam and ham. Just simply by reading the emails and the subject lines, I found that most spam emails have a lot of weird/uncommon punctuation (sometimes all in a row)--(i.e. $, ###, %, {}, <>, etc.). These details were very helpful in differentiating between emails. </li>

	<li> In my process of looking for better features, I realized that certain features didn't improve the accuracy rate as much as I had hoped. For example, if you use the number of "<>" in an email message as a feature, the improvement wasn't significant at all, and in fact, could sometimes lower the accuracy rate (depending on the combination of features). But expanding on this, I was able to determine a feature that did work--the number of special characters (i.e. sum of the total number of $, #, <>, {}, etc. charactes that appear in the message string). Though, what worked especially well, (more so than just the sum of special characters) was including a threshold value. For instance, if the sum of special characters was greater than 70 in the email message, it was more likely to be a spam email. </li>

	<li> What I found that was the most surprising in my search for good features was that the thresholding method (i.e. the total number of "something" is greater than/less than "some number") works really well in improving the accuracy rate of my score. For instance, adding the feature "greater than thresh" (i.e. whether the total length of email message was greater than 3000) drastically improved my score by around 2-3% as compared to simply using the total length of the email message. This is the same for several other features, such as the special character theshold, and the capital letters threshold. Thresholding the features of certain counts/percentages changes the feature into an indicator value of either 0 or 1, and adds more value to my feature matrix than simply the count/percentage feature itself. </li>

</ol>


**Citations**

<li> header image taken from google </li>
<li> Rights/credits to the creation of this assignment goes to UC Berkeley's DS100 class (Spring 2019) (link here)</li>