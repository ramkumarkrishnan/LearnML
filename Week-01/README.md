# Week 01 - Machine Learning

https://www.coursera.org/learn/machine-learning/home/week/1

Content: This week, we introduce the core idea of teaching a computer to learn concepts using data—without being explicitly programmed.  We are going to start by covering linear regression with one variable. Linear regression predicts a real-valued output based on an input value. We discuss the application of linear regression to housing price prediction, present the notion of a cost function, and introduce the gradient descent method for learning.  We’ll also have optional lessons that provide a refresher on linear algebra concepts. Basic understanding of linear algebra is necessary for the rest of the course, especially as we begin to cover models with multiple variables. If you feel confident in your understanding of linear algebra, feel free to take a break or help other students out in the forums.

What is Machine Learning? The science of getting computers to learn, without being explicitly programmed.

ML Examples:
- Google Search
- Apples Photo app recognizes your pictures
- Email - separate spam from non-spam
- Robots - make robots watch you do the work and learn from that.

ML Honor Code: Form groups to discuss, but do not show your code, or copy a classmate's code.

Questions:
- What is Octave work area?

Welcome - Introduction
- Learning algorithm - mimics how the human brain works.
- Knowing the math and algorithms is not sufficient, work on problems that you care about

- Database mining - 
-- Web Click/Clickstream data - understand data
-- Medical records -> medical knowledge
-- Computation biology -> Gene sequences
-- Engineering

- Apps that cannot be programmed by hand
-- Autonomous helicopter
-- Handwriting recognition
-- Natural language processing
-- Computer Visiono

- Self customizing programs
-- million programs for millionn users

- Understanding human learning
-- Brain
-- Unfulfilled skillset

**Next lesson - ML algorithms**
Learning algorithms in Practice
How to design ML in systems
Supervised and Unsupervised Learning - when to use them

Notes from Slides:
Two definitions of Machine Learning are offered. Arthur Samuel described it as: "the field of study that gives computers the ability to learn without being explicitly programmed." This is an older, informal definition.

Tom Mitchell provides a more modern definition: "A computer program is said to learn from experience E with respect to some class of tasks T and performance measure P, if its performance at tasks in T, as measured by P, improves with experience E."

Example: playing checkers.

E = the experience of playing many games of checkers

T = the task of playing checkers.

P = the probability that the program will win the next game.

In general, any machine learning problem can be assigned to one of two broad classifications:

Supervised learning and Unsupervised learning.
Examples of supervised learning:
- Predict the sale price of the house - Data provided: sq foot on the X axis, price on the Y axis, plot the graph with data, and then try to fit a curve. This is example of REGRESSION.
- Breast cancer: Age and Tumor size, Clump thickness, uniformity of cell size, unfiformity of cell shape - infinite number of features
- Mathematical trick that allows the computer to deal with a lot of features.
- Classification - discrete value (has or has not breast cancer)

In this video, I'm going to define what is probably the most common type of Machine Learning problem, which is Supervised Learning. I'll define Supervised Learning more formally later, but it's probably best to explain or start with an example of what it is, and we'll do the formal definition later. Let's say you want to predict housing prices. A while back a student collected data sets from the City of Portland, Oregon, and let's say you plot the data set and it looks like this. Here on the horizontal axis, the size of different houses in square feet, and on the vertical axis, the price of different houses in thousands of dollars. So, given this data, let's say you have a friend who owns a house that is say 750 square feet, and they are hoping to sell the house, and they want to know how much they can get for the house. So, how can the learning algorithm help you? One thing a learning algorithm might be want to do is put a straight line through the data, also fit a straight line to the data. Based on that, it looks like maybe their house can be sold for maybe about $150,000. But maybe this isn't the only learning algorithm you can use, and there might be a better one. For example, instead of fitting a straight line to the data, we might decide that it's better to fit a quadratic function, or a second-order polynomial to this data. If you do that and make a prediction here, then it looks like, well, maybe they can sell the house for closer to $200,000. One of the things we'll talk about later is how to choose, and how to decide, do you want to fit a straight line to the data? Or do you want to fit a quadratic function to the data? There's no fair picking whichever one gives your friend the better house to sell. But each of these would be a fine example of a learning algorithm. So, this is an example of a Supervised Learning algorithm. The term Supervised Learning refers to the fact that we gave the algorithm a data set in which the, called, "right answers" were given. That is we gave it a data set of houses in which for every example in this data set, we told it what is the right price. So, what was the actual price that that house sold for, and the task of the algorithm was to just produce more of these right answers such as for this new house that your friend may be trying to sell. To define a bit more terminology, this is also called a regression problem. By regression problem, I mean we're trying to predict a continuous valued output. Namely the price. So technically, I guess prices can be rounded off to the nearest cent. So, maybe prices are actually discrete value. But usually, we think of the price of a house as a real number, as a scalar value, as a continuous value number, and the term regression refers to the fact that we're trying to predict the sort of continuous values attribute. Here's another Supervised Learning examples. Some friends and I were actually working on this earlier. Let's say you want to look at medical records and try to predict of a breast cancer as malignant or benign. If someone discovers a breast tumor, a lump in their breast, a malignant tumor is a tumor that is harmful and dangerous, and a benign tumor is a tumor that is harmless. So obviously, people care a lot about this. Let's see collected data set. Suppose you are in your dataset, you have on your horizontal axis the size of the tumor, and on the vertical axis, I'm going to plot one or zero, yes or no, whether or not these are examples of tumors we've seen before are malignant, which is one, or zero or not malignant or benign. So, let's say your dataset looks like this, where we saw a tumor of this size that turned out to be benign, one of this size, one of this size, and so on. Sadly, we also saw a few malignant tumors cell, one of that size, one of that size, one of that size, so on. So in this example, I have five examples of benign tumors shown down here, and five examples of malignant tumors shown with a vertical axis value of one. Let's say a friend who tragically has a breast tumor, and let's say her breast tumor size is maybe somewhere around this value, the Machine Learning question is, can you estimate what is the probability, what's the chance that a tumor as malignant versus benign? To introduce a bit more terminology, this is an example of a classification problem. The term classification refers to the fact, that here, we're trying to predict a discrete value output zero or one, malignant or benign. It turns out that in classification problems, sometimes you can have more than two possible values for the output. As a concrete example, maybe there are three types of breast cancers. So, you may try to predict a discrete value output zero, one, two, or three, where zero may mean benign, benign tumor, so no cancer, and one may mean type one cancer, maybe three types of cancer, whatever type one means, and two mean a second type of cancer, and three may mean a third type of cancer. But this will also be a classification problem because this are the discrete value set of output corresponding to you're no cancer, or cancer type one, or cancer type two, or cancer types three. In classification problems, there is another way to plot this data. Let me show you what I mean. I'm going to use a slightly different set of symbols to plot this data. So, if tumor size is going to be the attribute that I'm going to use to predict malignancy or benignness, I can also draw my data like this. I'm going to use different symbols to denote my benign and malignant, or my negative and positive examples. So, instead of drawing crosses, I'm now going to draw O's for the benign tumors, like so, and I'm going to keep using X's to denote my malignant tumors. I hope this figure makes sense. All I did was I took my data set on top, and I just mapped it down to this real line like so, and started to use different symbols, circles and crosses to denote malignant versus benign examples. Now, in this example, we use only one feature or one attribute, namely the tumor size in order to predict whether a tumor is malignant or benign. In other machine learning problems, when we have more than one feature or more than one attribute. Here's an example, let's say that instead of just knowing the tumor size, we know both the age of the patients and the tumor size. In that case, maybe your data set would look like this, where I may have a set of patients with those ages, and that tumor size, and they look like this, and different set of patients that look a little different, whose tumors turn out to be malignant as denoted by the crosses. So, let's say you have a friend who tragically has a tumor, and maybe their tumor size and age falls around there. So, given a data set like this, what the learning algorithm might do is fit a straight line to the data to try to separate out the malignant tumors from the benign ones, and so the learning algorithm may decide to put a straight line like that to separate out the two causes of tumors. With this, hopefully we can decide that your friend's tumor is more likely, if it's over there that hopefully your learning algorithm will say that your friend's tumor falls on this benign side and is therefore more likely to be benign than malignant. In this example, we had two features namely, the age of the patient and the size of the tumor. In other Machine Learning problems, we will often have more features. My friends that worked on this problem actually used other features like these, which is clump thickness, clump thickness of the breast tumor, uniformity of cell size of the tumor, uniformity of cell shape the tumor, and so on, and other features as well. It turns out one of the most interesting learning algorithms that we'll see in this course, as the learning algorithm that can deal with not just two, or three, or five features, but an infinite number of features. On this slide, I've listed a total of five different features. Two on the axis and three more up here. But it turns out that for some learning problems what you really want is not to use like three or five features, but instead you want to use an infinite number of features, an infinite number of attributes, so that your learning algorithm has lots of attributes, or features, or cues with which to make those predictions. So, how do you deal with an infinite number of features? How do you even store an infinite number of things in the computer when your computer is going to run out of memory? It turns out that when we talk about an algorithm called the Support Vector Machine, there will be a neat mathematical trick that will allow a computer to deal with an infinite number of features. Imagine that I didn't just write down two features here and three features on the right, but imagine that I wrote down an infinitely long list. I just kept writing more and more features, like an infinitely long list of features. It turns out we will come up with an algorithm that can deal with that. So, just to recap, in this course, we'll talk about Supervised Learning, and the idea is that in Supervised Learning, in every example in our data set, we are told what is the correct answer that we would have quite liked the algorithms have predicted on that example. Such as the price of the house, or whether a tumor is malignant or benign. We also talked about the regression problem, and by regression that means that our goal is to predict a continuous valued output. We talked about the classification problem where the goal is to predict a discrete value output. Just a quick wrap up question. Suppose you're running a company and you want to develop learning algorithms to address each of two problems. In the first problem, you have a large inventory of identical items. So, imagine that you have thousands of copies of some identical items to sell, and you want to predict how many of these items you sell over the next three months. In the second problem, problem two, you have lots of users, and you want to write software to examine each individual of your customer's accounts, so each one of your customer's accounts. For each account, decide whether or not the account has been hacked or compromised. So, for each of these problems, should they be treated as a classification problem or as a regression problem? When the video pauses, please use your mouse to select whichever of these four options on the left you think is the correct answer.
Play video starting at 11 minutes 19 seconds and follow transcript11:19
So hopefully, you got that. This is the answer. For problem one, I would treat this as a regression problem because if I have thousands of items, well, I would probably just treat this as a real value, as a continuous value. Therefore, the number of items I sell as a continuous value. For the second problem, I would treat that as a classification problem, because I might say set the value I want to predict with zero to denote the account has not been hacked, and set the value one to denote an account that has been hacked into. So, just like your breast cancers where zero is benign, one is malignant. So, I might set this be zero or one depending on whether it's been hacked, and have an algorithm try to predict each one of these two discrete values. Because there's a small number of discrete values, I would therefore treat it as a classification problem. So, that's it for Supervised Learning. In the next video, I'll talk about Unsupervised Learning, which is the other major category of learning algorithm.

UNSUPERVISED LEARNING
- Regression - continuous value (price of the house)

No right answer - see if you can see structure them

Uses:
- Clustering - Google News
- Genomics - DNA micro array data (how much they do or do not have a gene)
Here is a bunch of data; I don't know what is in the data; can you automatically cluster - we are not giving the algorithm the right answer, 
- Large computer clusters - make DC more efficiently
- Which are cohesive groups of friends from social network analysis
- Market segmentation - group customers into different market segments.
- Astronomical analysis

Cocktail party problem:
- overlapping voices - everyone is talking at the same time
- microphone 1 and 2 - mics are recording two voices
- Person1 \   Mic 1  --> Algorithm -> Separate out the mixed audio (English)
- Person2 /   Mic 2  -->           -> Separate out the mixed audio (Spanish)
- The algorithm figures out there are two audio sources, and then separates them output.

Unsupervised Learning
Unsupervised learning allows us to approach problems with little or no idea what our results should look like. We can derive structure from data where we don't necessarily know the effect of the variables.  We can derive this structure by clustering the data based on relationships among the variables in the data.  With unsupervised learning there is no feedback based on the prediction results.

Example:

Clustering: Take a collection of 1,000,000 different genes, and find a way to automatically group these genes into groups that are somehow similar or related by different variables, such as lifespan, location, roles, and so on.

Non-clustering: The "Cocktail Party Algorithm", allows you to find structure in a chaotic environment. (i.e. identifying individual voices and music from a mesh of sounds at a cocktail party).

Algorithm to come with one line of code: 
[W, s, v] = SVD ((repmat(sum(x.*x, 1), size(x,1),1).*x)*x');

Prototype using Octave
You learn much faster with Octove as your programming env to prototype.  Then you move this to Java or C++.  We decided to go with Python after going with Octane. Using:
https://github.com/dibgerge/ml-coursera-python-assignments

Univariate Linear Regression - single variable, linear function, regression (since the output is continuous).

**Cost Function**

In this video we'll define something called the cost function, this will let us figure out how to fit the best possible straight line to our data.
Play video starting at 10 seconds and follow transcript0:10
In linear progression, we have a training set that I showed here remember on notation M was the number of training examples, so maybe m equals 47. And the form of our hypothesis,
Play video starting at 22 seconds and follow transcript0:22
which we use to make predictions is this linear function.
Play video starting at 26 seconds and follow transcript0:26
To introduce a little bit more terminology, these theta zero and theta one, they stabilize what I call the parameters of the model. And what we're going to do in this video is talk about how to go about choosing these two parameter values, theta 0 and theta 1. With different choices of the parameter's theta 0 and theta 1, we get different hypothesis, different hypothesis functions. I know some of you will probably be already familiar with what I am going to do on the slide, but just for review, here are a few examples. If theta 0 is 1.5 and theta 1 is 0, then the hypothesis function will look like this.
Play video starting at 1 minute 10 seconds and follow transcript1:10
Because your hypothesis function will be h of x equals 1.5 plus 0 times x which is this constant value function which is phat at 1.5. If theta0 = 0, theta1 = 0.5, then the hypothesis will look like this, and it should pass through this point 2,1 so that you now have h(x). Or really h of theta(x), but sometimes I'll just omit theta for brevity. So h(x) will be equal to just 0.5 times x, which looks like that. And finally, if theta zero equals one, and theta one equals 0.5, then we end up with a hypothesis that looks like this. Let's see, it should pass through the two-two point. Like so, and this is my new vector of x, or my new h subscript theta of x. Whatever way you remember, I said that this is h subscript theta of x, but that's a shorthand, sometimes I'll just write this as h of x.
Play video starting at 2 minutes 13 seconds and follow transcript2:13
In linear regression, we have a training set, like maybe the one I've plotted here. What we want to do, is come up with values for the parameters theta zero and theta one so that the straight line we get out of this, corresponds to a straight line that somehow fits the data well, like maybe that line over there.
Play video starting at 2 minutes 34 seconds and follow transcript2:34
So, how do we come up with values, theta zero, theta one, that corresponds to a good fit to the data?
Play video starting at 2 minutes 42 seconds and follow transcript2:42
The idea is we get to choose our parameters theta 0, theta 1 so that h of x, meaning the value we predict on input x, that this is at least close to the values y for the examples in our training set, for our training examples. So in our training set, we've given a number of examples where we know X decides the wholes and we know the actual price is was sold for. So, let's try to choose values for the parameters so that, at least in the training set, given the X in the training set we make reason of the active predictions for the Y values. Let's formalize this. So linear regression, what we're going to do is, I'm going to want to solve a minimization problem. So I'll write minimize over theta0 theta1. And I want this to be small, right? I want the difference between h(x) and y to be small. And one thing I might do is try to minimize the square difference between the output of the hypothesis and the actual price of a house. Okay. So lets find some details. You remember that I was using the notation (x(i),y(i)) to represent the ith training example. So what I want really is to sum over my training set, something i = 1 to m, of the square difference between, this is the prediction of my hypothesis when it is input to size of house number i.
Play video starting at 4 minutes 22 seconds and follow transcript4:22
Right? Minus the actual price that house number I was sold for, and I want to minimize the sum of my training set, sum from I equals one through M, of the difference of this squared error, the square difference between the predicted price of a house, and the price that it was actually sold for. And just remind you of notation, m here was the size of my training set right? So my m there is my number of training examples. Right that hash sign is the abbreviation for number of training examples, okay? And to make some of our, make the math a little bit easier, I'm going to actually look at we are 1 over m times that so let's try to minimize my average minimize one over 2m. Putting the 2 at the constant one half in front, it may just sound the math probably easier so minimizing one-half of something, right, should give you the same values of the process, theta 0 theta 1, as minimizing that function.
Play video starting at 5 minutes 24 seconds and follow transcript5:24
And just to be sure, this equation is clear, right? This expression in here, h subscript theta(x), this is our usual, right?
Play video starting at 5 minutes 37 seconds and follow transcript5:37
That is equal to this plus theta one xi. And this notation, minimize over theta 0 theta 1, this means you'll find me the values of theta 0 and theta 1 that causes this expression to be minimized and this expression depends on theta 0 and theta 1, okay? So just a recap. We're closing this problem as, find me the values of theta zero and theta one so that the average, the 1 over the 2m, times the sum of square errors between my predictions on the training set minus the actual values of the houses on the training set is minimized. So this is going to be my overall objective function for linear regression.
Play video starting at 6 minutes 22 seconds and follow transcript6:22
And just to rewrite this out a little bit more cleanly, what I'm going to do is, by convention we usually define a cost function,
Play video starting at 6 minutes 31 seconds and follow transcript6:31
which is going to be exactly this, that formula I have up here.
Play video starting at 6 minutes 37 seconds and follow transcript6:37
And what I want to do is minimize over theta0 and theta1. My function j(theta0, theta1). Just write this out.
Play video starting at 6 minutes 53 seconds and follow transcript6:53
This is my cost function.
Play video starting at 6 minutes 59 seconds and follow transcript6:59
So, this cost function is also called the squared error function.
Play video starting at 7 minutes 6 seconds and follow transcript7:06
When sometimes called the squared error cost function and it turns out that why do we take the squares of the erros. It turns out that these squared error cost function is a reasonable choice and works well for problems for most regression programs. There are other cost functions that will work pretty well. But the square cost function is probably the most commonly used one for regression problems. Later in this class we'll talk about alternative cost functions as well, but this choice that we just had should be a pretty reasonable thing to try for most linear regression problems.
Play video starting at 7 minutes 42 seconds and follow transcript7:42
Okay. So that's the cost function.
Play video starting at 7 minutes 45 seconds and follow transcript7:45
So far we've just seen a mathematical definition of this cost function. In case this function j of theta zero, theta one. In case this function seems a little bit abstract, and you still don't have a good sense of what it's doing, in the next video, in the next couple videos, I'm actually going to go a little bit deeper into what the cause function "J" is doing and try to give you better intuition about what is computing and why we want to use it...


GRADIENT DESCENT:
Gradient Descent is an optimization algorithm for finding a local minimum of a differentiable function. Gradient descent is simply used to find the values of a function's parameters (coefficients) that minimize a cost function as far as possible.Jun 16, 2019
https://builtin.com/data-science/gradient-descent

https://hackernoon.com/how-to-interpret-a-contour-plot-a617d45f91ba
