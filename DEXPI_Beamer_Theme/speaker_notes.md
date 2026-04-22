# Speaker Notes - Stress Detection Using Wearable Sensors
**STT881 Project Presentation**

---

## Slide 1: Title Slide

Good morning everyone! Today I'll be presenting our STT881 project on stress detection using wearable sensors. So basically what we did was use physiological data from wearable devices to classify different states of the human body - specifically whether someone is in an aerobic state, anaerobic state, or experiencing stress. This is a really cool application of machine learning because it combines real-world data collection with practical health monitoring applications. We'll walk through our entire pipeline today, from the raw sensor data all the way to our final model results, and I think you'll see some interesting patterns in how different machine learning algorithms handle this type of physiological data. Let's get started!

---

## Slide 2: Motivation and Objective

So first, let's talk about why this project matters. Chronic stress is actually a huge problem in modern society - it affects both mental and physical health in really significant ways. We're talking about increased risk of heart disease, depression, anxiety, all sorts of health issues. The traditional way of measuring stress is usually self-reported questionnaires, which are subjective and not real-time. But here's where wearable devices come in - they can provide continuous, non-invasive monitoring of physiological signals throughout the day. This means we could potentially detect stress as it's happening, which enables early intervention before it becomes a chronic problem.

So our specific goal for this project was to classify three different physiological states using data from wearable sensors. We had data from 100 participants, and we were trying to distinguish between AEROBIC states - that's like steady cardio exercise, ANAEROBIC states - that's high-intensity exercise, and STRESS states - which were induced in a controlled lab setting. The key challenge here is that all three of these states affect your body in similar ways - they all increase heart rate, they all affect skin conductance - so distinguishing between them using just physiological signals is actually quite difficult. That's what makes this an interesting machine learning problem.

---

## Slide 3: Data - Wearable Sensor Dataset

Let me tell you about our dataset. We had data from 100 participants total, which were pretty evenly distributed across the three classes - 31 in the AEROBIC group, 32 in ANAEROBIC, and 37 in the STRESS group. Each participant wore a wearable device for anywhere from 28 to 60 minutes while they were put through different protocols to induce these states.

Now, the really cool part is what the sensors were measuring. We had four different physiological signals being recorded simultaneously. First, EDA - that's electrodermal activity, basically measuring skin conductance at 4 Hertz. When you're stressed or exercising, your skin conductance changes because of sweat gland activity. Second, we had heart rate measured at 1 Hz - pretty straightforward, but super important for distinguishing exercise from stress. Third was skin temperature, also at 4 Hz - temperature can increase with exercise but sometimes drops during acute stress. And finally, we had 3-axis accelerometer data at 32 Hz, which captures all the body movement.

So you can imagine, for a 30-minute session, we're talking about thousands and thousands of data points per person per signal. The challenge is going to be how to turn all that high-frequency time series data into something we can feed into traditional machine learning models. We'll get to that in the feature engineering section.

---

## Slide 4: Exploratory Data Analysis

So before jumping into modeling, we did some exploratory data analysis to understand what our data actually looks like. The top figure here shows time series examples from all four sensors for each of the three states. If you look at these plots, you can start to see some patterns. For instance, in the ANAEROBIC state, you can see really high heart rate values and lots of accelerometer activity because people are doing high-intensity exercise. The AEROBIC state shows more moderate, steady signals. And the STRESS state is interesting - you might see elevated EDA and heart rate, but much less movement in the accelerometer.

The bottom figure is a correlation heatmap showing how the different signals relate to each other. What's interesting here is that some signals are correlated in certain states but not others. For example, accelerometer magnitude and heart rate might be highly correlated during exercise states but not during stress. This actually suggests that these signals contain complementary information, which is good news for our classification task. It means using all four signals together should give us better performance than using any one signal alone.

This EDA really helped us understand that yes, there are distinguishable patterns between these states, but they're subtle and we're going to need good preprocessing and the right model to capture them.

---

## Slide 5: Data Preprocessing

Alright, so preprocessing was a critical step for us. Raw sensor data is noisy - you've got measurement errors, you've got motion artifacts, you've got all sorts of issues. So we built a signal processing pipeline specific to each sensor type.

For EDA, we did outlier removal first because sometimes the sensors give you completely unrealistic values, then we applied a low-pass Butterworth filter at 1 Hz to remove high-frequency noise, and finally normalized the signal. For heart rate, we removed outliers and normalized - pretty straightforward because the sensor already does some processing. For temperature, we used z-score normalization because temperature values are pretty stable and we mainly care about deviations from baseline. And for the accelerometer, we normalized each axis and then computed the magnitude - that's the square root of the sum of squares of the three axes - which gives us a single value representing overall movement intensity.

Now, a couple implementation details that matter: we used a Butterworth filter because it has a nice flat frequency response in the passband, and we used scipy's filtfilt function, which does zero-phase filtering. That means it filters the signal forward and backward so you don't get phase distortion, which is important when you're trying to align events across multiple signals. All of this was implemented with vectorized NumPy operations, which made it run pretty fast even though we're processing hundreds of thousands of data points. After this preprocessing, our signals were much cleaner and ready for feature extraction.

---

## Slide 6: Feature Engineering

So here's where we hit a major challenge. We've got these time series that are 7,000 to 14,000 samples long depending on how long the person wore the device. But traditional machine learning algorithms - like logistic regression, SVM, KNN - they need fixed-length feature vectors. They can't just take variable-length time series as input. So we needed to convert these long sequences into a compact, fixed-size representation.

Our solution was to extract statistical features from each signal. Specifically, we computed four features per signal: the mean, which captures the average level; the standard deviation, which captures variability; the median, which is robust to outliers; and the 95th percentile, which captures extreme values without being as sensitive to outliers as the max. We applied these four features to each of our four signals - EDA, heart rate, temperature, and accelerometer magnitude. So that gives us 4 times 4, which equals 16 features total for each participant.

Now, you might think 16 features is pretty small - and you're right, it is relatively compact. We're compressing thousands of data points into just 16 numbers. This means we're definitely losing some information - things like temporal patterns, frequency domain features, heart rate variability metrics - we're not capturing those with these simple statistics. But the advantage is these features are interpretable, they're easy to compute, and they give us a starting point. If we can get decent performance with just these 16 features, that tells us the task is feasible, and we can always add more sophisticated features later.

---

## Slide 7: Model Training

For the modeling phase, we wanted to test a diverse set of algorithms to see what works best for this problem. We trained five different models. First, logistic regression - that's our linear baseline model. Second, K-Nearest Neighbors with k equals 5 - this is a non-parametric method that classifies based on similar examples. Third, a decision tree - which learns hierarchical rules and can capture non-linear patterns. Fourth, Support Vector Machine with an RBF kernel - this is a powerful non-linear classifier that tries to find optimal decision boundaries. And fifth, a neural network with one hidden layer of 50 neurons - this can learn complex non-linear representations.

In terms of training configuration, we did a standard 80-20 train-test split, which gave us 80 participants for training and 20 for testing. We applied StandardScaler to all our features, which is important especially for algorithms like SVM and neural networks that are sensitive to feature scales. And we set a random seed of 42 for reproducibility - so if you run our code, you'll get the same results.

We didn't do any hyperparameter tuning for most models - we just used reasonable default values - because we wanted to compare the models fairly and see which algorithms are fundamentally better suited to this problem. The one exception is the neural network where we did try a few different architectures and settled on the single hidden layer with 50 neurons as a good balance.

---

## Slide 8: Results - Model Comparison

Okay, so here are our results! This figure shows the test accuracy for all five models, and there's a clear winner here - SVM achieved the best accuracy at 65 percent. Let me put that in context: if you were just randomly guessing among three classes, you'd get 33.3% accuracy. So we're doing nearly twice as good as random, which is decent but also shows this is a hard problem.

Looking at the other models, the neural network came in second at 60%, logistic regression got 55%, KNN got 50%, and decision tree was the worst at 40%. There's a really interesting pattern here: the non-linear models - SVM and neural network - significantly outperform the linear model, which is logistic regression. This tells us that the decision boundaries between these classes are not linear in this 16-dimensional feature space.

The fact that SVM outperforms the neural network is interesting too - it suggests that with our relatively small dataset of 80 training examples, SVM's margin-based approach is more effective than the neural network's gradient-based learning. The decision tree doing poorly is not that surprising - decision trees can overfit easily and they're not great with continuous numerical features like we have here. KNN's mediocre performance suggests that the classes might not cluster well in the feature space - similar feature vectors don't always have the same label. Overall, that 65% SVM accuracy is our best result, and it's what we'll focus on in the next slide.

---

## Slide 9: Best Model - SVM Performance

This slide digs deeper into our best model, the SVM. We've got a confusion matrix here and classification metrics for each class. The confusion matrix shows where the model is making correct predictions - that's the diagonal - and where it's making mistakes.

What I really like about this result is that the SVM shows relatively balanced performance across all three states. If you look at the precision and recall values, they're all in a similar range - we're not doing great on one class and terrible on another. ANAEROBIC tends to be the easiest to detect, which makes sense because it has very distinctive patterns - high heart rate, high movement, different temperature response. The model does confuse AEROBIC and STRESS sometimes, which is understandable because both can elevate heart rate and EDA without as much movement.

But overall, the SVM is doing a pretty good job of using the RBF kernel to create non-linear decision boundaries that separate these classes. The balanced performance is important because in a real-world application, you'd want your stress detector to work well for detecting stress specifically, not just be good at detecting exercise and then failing on stress. This tells us that our feature engineering captured meaningful information about all three states, and the SVM was able to leverage that information effectively.

---

## Slide 10: Key Findings

Let me summarize the key takeaways from this project. First and most importantly, SVM significantly outperforms other models. We got 65% accuracy with SVM compared to 60% for neural networks, 55% for logistic regression, 50% for KNN, and only 40% for decision trees. That's a 25 percentage point improvement from the worst to the best model, which is huge! The non-linear kernel in the SVM is really what makes the difference - it's able to capture complex patterns in how these physiological signals combine to indicate different states.

Second finding: we achieved balanced performance across all three physiological states. This wasn't a given - we could have ended up with a model that's great at detecting one state but terrible at the others. But the SVM shows good precision and recall for all three classes. ANAEROBIC is still the easiest to detect because of its distinctive high-intensity exercise signature, but we reduced the confusion between AEROBIC and STRESS compared to simpler models.

Third finding: model selection really, really matters. We saw a 25% absolute improvement in accuracy just by choosing the right algorithm. This highlights that with a fixed feature set, you can still get dramatically different performance by choosing an appropriate model. The simple statistical features we extracted work much better with non-linear models like SVM and neural networks than with linear models or decision trees. If we had just tried logistic regression and stopped there, we would have concluded the task is barely better than random. But by testing multiple models, we found that SVM can actually do quite well.

---

## Slide 11: Conclusion

So to wrap everything up, let me summarize what we accomplished and what we learned.

**In terms of accomplishments:** We successfully tested five different machine learning models on this physiological state classification problem. We built a complete end-to-end pipeline starting from raw sensor data - EDA, heart rate, temperature, accelerometer - all the way through preprocessing, feature engineering, model training, and evaluation. Our best model, the Support Vector Machine, achieved 65% accuracy, which is almost double the 33% random baseline. That's solid performance given we only used 16 simple statistical features.

**In terms of key lessons learned,** there are several important ones. On the positive side: model selection is absolutely critical - we saw a 25% improvement just from choosing the right algorithm. Non-linear models, particularly SVM and neural networks, significantly outperform linear models for this task, which tells us the patterns in physiological data are complex and non-linear. And our preprocessing pipeline worked - it enabled all models to function and extract meaningful patterns. On the challenging side: 65% accuracy is decent but not amazing - this is still a hard problem, and simple statistical features have limitations. We're definitely losing information by compressing time series into just mean, std, median, and percentile.

**Looking ahead to future work,** there are several exciting directions. We could extract more sophisticated features like heart rate variability metrics or frequency domain features using Fourier transforms. We could try ensemble methods that combine multiple models. We could explore deep learning approaches like LSTMs that can handle time series directly without feature engineering. And ideally, we'd want a larger dataset - 100 participants is decent for a class project but relatively small for building robust models.

But overall, I think this project shows that physiological state classification using wearable sensors is feasible, and with the right approach, we can build systems that could eventually help with real-time stress monitoring and health interventions. Thank you for listening, and I'm happy to take any questions!
