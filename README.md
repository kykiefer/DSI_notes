# DSI_notes
A collection of notes from G39 DSI

## Schedule

| Week | Date     | Topic |
|:----:|:---------|:------|
| 0    | 03/27/17 | [Python workshop](#week-0-python-workshop) |
| 1    | 04/03/17 | [Programming](#week-1-programming) |
| 2    | 04/10/17 | [Probability and Statistics](#week-2-probability-and-statistics) |
| 3    | 04/17/17 | [Linear Regression](#week-3-linear-regression) |
| 4    | 04/24/17 | [Supervised Learning](#week-4-supervised-learning) |
| 5    | 05/01/17 | [Special Topics](#week-5-special-topics) |
| 6    | 05/08/17 | [Unsupervised Learning](#week-6-unsupervised-learning) |
| -    | 05/15/17 | Break Week (Optional time-series) |
| 7    | 05/22/17 | [Big Data / Data Engineering](#week-7-big-data--data-engineering) |
| 8    | 05/29/17 | [Special Topics / Case Studies](#week-8-special-topics--case-studies) |
| 9    | 06/05/17 | [Project](#week-9-project) |
| 10   | 06/12/17 | [Project](#week-10-more-project) |
| 11   | 06/19/17 | [Project](#week-11-even-more-project) |
| 12   | 06/26/17 | [Interview Prep](#week-12-interview-prep) |

--

Please familiarize yourself with the following before getting started

   * [Pair Programming][2]
   * [Install Postgres](https://github.com/gSchool/dsi-curriculum/tree/master/notes/postgres_setup.md)

### Week 0: Python workshop
| Day                   | Main Topics    | Readings (suggested)                               | Repo           | Lead    | Deck   | Slides        |
|:----------------------|:---------------|:---------------------------------------------------|:--------------:|:--------|:------:|:-------------:|
| Monday<br>03/27/16    | basics         | [Beginner's Guide][0.1]                            | [Day1][0.01]   | Erich   | Brent  | -             |
| Tuesday<br>03/28/16   | dicts,sets     | [Data Structures][0.2]                             | [Day2][0.02]   | Erich   | Brent  | -             |
| Wednesday<br>03/29/16 | OOP            | [Object-oriented programming][0.3]                 | [Day3][0.03]   | Erich   | Brent  | -             |
| Thursday<br>03/30/16  | review         |                                                    | [Day4][0.04]   | Erich   | Brent  | -             |
| Friday<br>03/31/16    | *no class*     |                                                    |                | -       | -      | -             |

* [My Repo](https://github.com/kykiefer/python-workshop)

--

### Week 1: Programming
| Day                  | Main Topics        | Readings                                                                   | Repo                                     | Lead  | Deck  | Slides         | My Repo | Daily Slides |
|:---------------------|:-------------------|:---------------------------------------------------------------------------|:----------------------------------------:|:------|:-----:|:--------------:|:-----:|:-----:|
| Monday<br>04/03/16   | assessment,python |[Workflow][1]                                       |[Assessment][2.1]<br>[Python Intro][3]| Erich | Brent | [slides][3.1]            | | [Intro to Python](https://github.com/kykiefer/DSI_notes/blob/master/daily_slides/w1/intro-to-python-slides.pdf) |
| Tuesday<br>04/04/16  | OOP               |[Learn Python (ex 40-42)][3.2] (Extra 43)           |[OOP][8]                              | Erich | Steve| [slides][3.3]            | [OOP](https://github.com/kykiefer/OOP) | [OOP](https://github.com/kykiefer/DSI_notes/blob/master/daily_slides/w1/oop-slides.pdf) |
| Wednesday<br>04/05/16| SQL               |[SQLZOO (tutorial: 1-9)][10] <br> [Joins][12]       |[SQL][13]                             | Frank | Brent | [slides][13.1]           | [SQL](https://github.com/kykiefer/sql) | [SQL](https://github.com/kykiefer/DSI_notes/blob/master/daily_slides/w1/sql-afternoon.pdf) |
| Thursday<br>04/06/16 | Jupyter,pandas    |[Pandas [1][19],[2][21]<br>[Extra1][20],[Extra2][22]|[sql-python][25.1]<br>[pandas][23]    | Erich | Steve | [slides][23.1],[2][23.2] | [sql-python](https://github.com/kykiefer/sql-python)<br><br>[pandas](null) | [SQL-python](https://github.com/kykiefer/DSI_notes/blob/master/daily_slides/w1/sql-python-slides.pdf) |
| Friday<br>04/07/16   | plotting          |*no readings*                                       |[pandas-seaborn][24.1]                | Adam  | Brent | [slides][24.2]           | [pandas-seaborn](https://github.com/kykiefer/pandas-seaborn) | [plotting](https://github.com/kykiefer/DSI_notes/blob/master/daily_slides/w1/basic-plotting.pdf) |

--

### Week 2: Probability and Statistics
| Day                  | Main Topics        | Readings                                                                   | Repo                                     | Lead  | Deck  | Slides         | My Repo | Daily Slides |
|:---------------------|:-------------------|:---------------------------------------------------------------------------|:----------------------------------------:|:------|:-----:|:--------------:|:-----:|:-----:|
| Monday<br>04/10/16   | probability        | [Probability (1.2-4.3),1 hour][25]<br>[Simple stats with SciPy][25.01]     | [Assessment 2][28.0]<br>[Probability][28]| Frank | Steve | [slides][28.1] | [Probability](https://github.com/kykiefer/probability) | |
| Tuesday<br>04/11/16  | sampling,estimation| [Bootstrap][38.1]<br>[CLT][40.1]<br>CIs [1][38.0],[2][38.01]<br>[MLE][38.2]| [Sampling and Estimation][38]            | Brent | Adam  | [slides][38.3] | [Sampling and Estimation](https://github.com/kykiefer/estimation-sampling) | [estimation and sample](https://github.com/kykiefer/DSI_notes/blob/master/daily_slides/w2/eestimation-sampling-lecture.pdf) |
| Wednesday<br>04/12/16| hypothesis testing | [z-test VS t-test][39.1]<br>[Hypothesis Testing][39.2]                     | [Hypothesis Testing][39]                 | Frank | Steve | [slides][39.3] | [Hypothesis Testing](https://github.com/kykiefer/ab-testing) |
| Thursday<br>04/13/16 | power, Bayes       | [Power Analysis][39.4]<br>[Bayesian stats 1][40.0]                         | [Power Calculation and Bayes][40]        | Adam  | Brent | [slides][40.2] | [Power Calculation and Bayes](https://github.com/kykiefer/power-bayesian) |
| Friday<br>04/14/16   | multi-armed bandit | *no reading*                                                               | [Multi-armed bandit][44]                 | Adam  | Steve | [slides][40.5] | [Multi-arm Bandit](https://github.com/kykiefer/multi-armed-bandit) |

--

### Week 3: Linear Models
| Day                  | Main Topics        | Readings                                                                   | Repo                                     | Lead  | Deck  | Slides         | My Repo | Daily Slides |
|:---------------------|:-------------------|:---------------------------------------------------------------------------|:----------------------------------------:|:------|:-----:|:--------------:|:-----:|:-----:|
| Monday<br>04/17/17   | NumPy, Linear Reg. |[Linear Algebra][45]<br>[SL (3-3.2, pg 59-82)][47.1]<br>[Optional][45.1] |[Linear Algebra, EDA, Lin. Reg.][48]             | Frank      | Brent  |[slides][45.2] | [Linear Algebra](https://github.com/kykiefer/linear-algebra-eda) |
| Tuesday<br>04/18/17  | More Linear Reg.   |[ISLR (3.3-3.4, pg 82-104)][47.1]                                        |[Linear Regression 2][58]                        | Adam       | Steve  |[slides][45.3] | [Linear Regression](https://github.com/kykiefer/linear-regression) |
| Wednesday<br>04/19/17| Regularization, CV |[ISLR (5-5.1.4, pg 175-184)][47.1]<br>[SL (6.2, pg 214-228)][47.1]       |[Cross Validation & Regularization][54]          | Frank      | Brent  |[slides][54.1] | [Regularized Regression](https://github.com/kykiefer/regularized-regression) |
| Thursday<br>04/20/17 | ROC, Logistic Reg. |[ISLR (pg 127-137)][47.1]<br>[ML in action (pg 142-148)][MLIA]           |[Logistic Reg.][log-reg]                         | Adam       | Steve  |[slides][54.2] | [Logistic Regression](https://github.com/kykiefer/logistic-regression) |
| Friday<br>04/21/17   | Data Visualization |*no reading*                                                             |[Assessment 3][A3]<br>[Data Visualization][150.1]| Steve      | Adam   |[slides][150.2]| [Data Viz](https://github.com/kykiefer/data-viz-for-ds) |

#### Optional reading
   * Linear model selection and Regularization [ISLR (6.2, pg 203-214)][47.1]

--

### Week 4: Supervised Learning
| Day                  | Main Topics        | Readings                                                                   | Repo                                     | Lead  | Deck  | Slides         | My Repo | Daily Slides |
|:---------------------|:-------------------|:---------------------------------------------------------------------------|:----------------------------------------:|:------|:-----:|:--------------:|:-----:|:-----:|
| Monday<br>02/20/17    | Gradient Decent       |[ML in Action (ch 5, pg 83-90)][MLIA]<br>(optional 90-96)         |[Gradient Descent][52]                        | Adam  | Frank | [slides][54.3] | [Gradient Descent](https://github.com/kykiefer/gradient-descent) |
| Tuesday<br>04/25/17   | Decision Trees, KNN   |[ML in Action (pg 18-24,pg 37-48)][MLIA]<br>[Recursion][recursion]|[Decision Trees and KNNs][65]                 | Frank | Steve | [slides][65.1] | [Decision Trees and KNNs](https://github.com/kykiefer/non-parametric-learners) |
| Wednesday<br>04/26/17 |Bagging, Random Forests|[ISLR (8.1.2-8.2.2, pg 311-321)][47.1]                            |[Bagging & Random Forests][68]                | Brent | Frank | [slides][68.1] |[Bagging and Ranomd Forsets](https://github.com/kykiefer/random-forest) |
| Thursday<br>04/27/17  | Kernels, SVMs         |[ISLR (9-9.2, pg 337-349)][47.1]                                  |[SVMs and Kernels][71]                        | Adam  | Brent | [slides][71.2] |[SVMs and Kernels](https://github.com/kykiefer/svm) |
| Friday<br>04/28/17    | *case study*          | -                                                                |[Regression Case Study][71.5]                 | TBD   | -     | -              | [Regression Case Study](https://github.com/kykiefer/regression-case-study) |

#### Optional reading

   * For a visual explanation of decision trees [Decision Tree Visual Explanation][47.2]
   * Recursion - [ISLR (8.1 pg 303-316)][47.1]
   * SVM's - [ISLR (9.3-9.3.2, pg 349-353)][47.1]
   * Boosting - [Elements of Stats Learning (10-10.6, pg 337-350)][esl]
   * For even more rigor with Gradient Decent see [Andrew Ng Notes (p. 1-7, 16-19)][51]

--

### Week 5: Special Topics
| Day                  | Main Topics        | Readings                                                                   | Repo                                     | Lead  | Deck  | Slides         | My Repo | Daily Slides |
|:---------------------|:-------------------|:---------------------------------------------------------------------------|:----------------------------------------:|:------|:-----:|:--------------:|:-----:|:-----:|
| Monday<br>05/01/17   |boosting,assessment | [MMD ch 2][mmd2] (pg 21-41)                                        | [Boosting][boosting]<br>[Assessment 4][A4]          | Frank  | Brent | [1][71.3]      | [Boosting](https://github.com/kykiefer/boosting) |
| Tuesday<br>05/02/17  |web-scraping,MongoDB| [Precourse - Web][75][Web Scaping][76][MongoDB][76.1]              | [Web Scraping][77]                                  | Steve  | Adam  | [1][77.1]      | [Web Scraping](https://github.com/kykiefer/web_scraping) |
| Wednesday<br>05/03/17|Neural Nets         | [Oxford Deep Lrn course][71.7]<br>[Galaxy classification][71.8]    | [Neural-networks][nn]                               | Frank  | Brent | [1][71.6]      | [Neural Networks](https://github.com/kykiefer/mlp_cnn_rnn) |
| Thursday<br>05/04/17 |Naive Bayes, NLP    | [Text feature extraction (tf-idf)][tfidf1], [NLP][NLP](pg 107-108) | [NLP][84]                                           | Adam   | Frank | [1][84.1]      | [NLP](https://github.com/kykiefer/nlp |
| Friday<br>05/05/17   | clustering         | [ISLR (pg 385-400)][47.1]                                          | [KMeans and Hierarchical Clustering][104]           | Frank  | Steve | [1][104.1]     | [ 385-400)][47.1]                                          | [KMeans and Hierarchical Clustering](https://github.com/kykiefer/clustering) |

#### Optional reading

   * NLP - [Natural Language Processing with Python][NLP] (ch 3, pg 79-122)
   * TF - IDF [1][tfidf2],[2][tfidf3]

--

### Week 6: Unsupervised Learning
| Day                  | Main Topics        | Readings                                                                   | Repo                                     | Lead  | Deck  | Slides         | My Repo | Daily Slides |
|:---------------------|:-------------------|:---------------------------------------------------------------------------|:----------------------------------------:|:------|:-----:|:--------------:|:-----:|:-----:|
| Monday<br>05/08/17   | dimension reduction    |[ML in Action][MLIA] (ch 13-14.3 pg 269-286)<br>[ISLR (pg 374-385)][47.1] |[Dimensionality Reduction][107]          | Steve | Adam  |[1][107.1]           |
| Tuesday<br>05/09/17  | NMF                    |[NMF in Python][nmf-reading]                                              |[Assessment 5][A5]<br><br>[NMF][nmf]     | Adam  | Brent |[1][103.1]           |
| Wednesday<br>05/10/17| Graph theory           |[Social Network Analysis][sna] (pg 19-38),  [MMD][mmd10](pg 343-356)      |[Graph Theory][graphs]                   | Adam  | Brent |[1][122]             |
| Thursday<br>05/11/17 | profit curves, map-red |[DSFB][DSBus] (pg 194-203, 212-214) <br> [ISLR (8.2.3, pg 321-324)][47.1] |[Profit Curves][profit]<br>[MR][129]     | Frank | Adam  |[1][122.3], [2][71.4]|
| Friday<br>05/12/17   | **case study**         | **no reading**                                                           | [Churn Case Study][200]                 | TBD   | -     | -                   |

#### Optional reading

   * [Generators][generators]<br>[FP][funcprog]
   * [Mining Massive Datasets ch 11][mmd11]
   * [Mining Massive Datasets ch 9][mmd9]
   * [Mining Massive Datasets ch 9][mmd9] (ch 9.4, pg 328-337)
   * [Recommender systems: from algorithms to user experience][rec-paper]
   * Business analytics - [Data Science for Business][DSBus] - remainder of ch 7-8, pg 187-232

### Break Week
| Day                  | Main Topics        | Readings                                                      | Repo                                         | Lead       | Deck     | Slides   |
|:---------------------|:-------------------|:--------------------------------------------------------------|:--------------------------------------------:|:-----------|:--------:|:--------:|
| Monday<br>05/15/17   | *off*              |*no class*                                                     | -                                            | -          | -        | -        |
| Tuesday<br>05/16/17  | *off*              |*no class*                                                     | -                                            | -          | -        | -        |
| Wednesday<br>05/17/17| *off*              |*no class*                                                     | -                                            | -          | -        | -        |
| Thursday<br>05/18/17 | *off*              |*no class*                                                     | -                                            | -          | -        | -        |
| Friday<br>05/19/17   | *off*              |*no class*                                                     | -                                            | -          | -        | -        |

#### Optional break week work time-series
   * [Forecasting][hyndman] (Ch. 1, 2, & 6-8)
   * [Time Series][timeseries] (ch 1-3)[ARIMA][arima]|
   * [Time Series][time-series]
   * [slides][103.6]

### Week 7: Big Data / Data Engineering
| Day                  | Main Topics        | Readings                                                                   | Repo                                     | Lead  | Deck  | Slides         | My Repo | Daily Slides |
|:---------------------|:-------------------|:---------------------------------------------------------------------------|:----------------------------------------:|:------|:-----:|:--------------:|:-----:|:-----:|
| Monday<br>05/22/17   | AWS, speedy computing |[Multiprocessing][multiproc-python]<br>[Parallel][parallel-intro]     | [AWS and Parallelization][hp-python]   | DSRs     | DSRs  | [slides][122.2]|
| Tuesday<br>05/23/17  | Spark (proposals due) |[Learning Spark][LearningSpark] (ch 1-2, pg 1-22)                     | [Spark][spark]                         | Brent    | Steve | [slides][122.4]|
| Wednesday<br>05/24/17| Spark on AWS          |[Spark on AWS][131.8]<br>[Learning Spark][LearningSpark] (pg 135-139) | [SparkSQL and Spark on AWS][spark-aws] | Frank    | Brent | [slides][122.5]|
| Thursday<br>05/25/17 | Recommendors          |[ML in Action][MLIA] (pg 286-295)  (especially 9.1, 9.3, 9.5)         | [Recommendation Systems][118]          | Adam     | Frank | [slides][103.4]|
| Friday<br>05/26/17   | *case study*          |[MFTFRS][mftr]                                                        | [Recommender Case Study][rec2]         | Adam     | -     | [slides][103.5]|

#### Optional reading

   * [Threading][threading]
   * [Learning Spark][LearningSpark] (ch 11: MLlib, pg 183-212)
   * [Data-Intensive Text Processing with MapReduce][mrbook] (ch 1-3 pg 1-69)

--

### Week 8: Special Topics / Case Studies
| Day                  | Main Topics        | Readings                                                                   | Repo                                     | Lead  | Deck  | Slides         | My Repo | Daily Slides |
|:---------------------|:-------------------|:---------------------------------------------------------------------------|:----------------------------------------:|:------|:-----:|:--------------:|:-----:|:-----:|
| Monday<br>05/29/17   | *off*                 |*no class*                                                                       | -                                 | -         | -         | -        |
| Tuesday<br>05/30/17  | Flask, data products  | [Setup Flask][132.1] (5min)<br>[Flask Tutorials][132.2]<br>[Get vs Post][150.3] | [Data Products][132.0]            | Frank     | Frank | [slides][150.4]|
| Wednesday<br>05/31/17| *case study*          | *Final project proposals due at 9:30am*                                         | [Fraud Detection Case Study][135] | Steve     | -         |                |
| Thursday<br>06/01/17 | *case study*          | Fraud Detection Case Study (continued)                                          | [Fraud Detection Case Study][135] | Brent     |          |                |
| Friday<br>06/02/17   | final assessment      | CAREER SERVICES, Agile/Scrum lecture                                            | [Final Assessment][131.1]         | Adam      | -         |                |


Notes
####
   * For the Flask setup do as many as you see fit, dont worry about setting up the virtual environment
   * [Optional] [Data Science for Business][DSBus] (ch 1-2, 14, pg 1-42, 331-346)
   * [Optional] [Data Science Use Cases][150.5]

--

### Week 9: Project
| Day                  | Main Topics        | During day                                                          | Repo                                         | Lead       |On Deck     | Slides       |
|:---------------------|:-------------------|:--------------------------------------------------------------------|:--------------------------------------------:|:-----------|:----------:|:------------:|
| Monday<br>06/05/17   | *off*              | *no class*                                                          | -                                            | -          | -          | -            |
| Tuesday<br>06/06/17  | Projects           | Daily scrum meeting                                                 | [project-proposals][proj]                    | -          | -          | -            |
| Wednesday<br>06/07/17| Projects           | Daily scrum meeting<br>Resume lecture (career services)             | -                                            | -          | -          | -            |
| Thursday<br>06/08/17 | Projects           | Daily scrum meeting                                                 | -                                            | -          | -          | -            |
| Friday<br>06/09/17   | Projects           | Daily scrum meeting<br>README lecture                               | -                                            | -          | -          | -            |

--

### Week 10: Project
| Day                  | Main Topics        | During day                                                          | Repo                                         | Lead       |On Deck     | Slides       |
|:---------------------|:-------------------|:--------------------------------------------------------------------|:--------------------------------------------:|:-----------|:----------:|:------------:|
| Monday<br>06/12/17   | Projects           | Daily scrum meeting                                                 | -                                            | -          | -          | -            |
| Tuesday<br>06/13/17  | Projects           | Daily scrum meeting                                                 | -                                            | -          | -          | -            |
| Wednesday<br>06/14/17| Projects           | Daily scrum meeting                                                 | -                                            | -          | -          | -            |
| Thursday<br>06/15/17 | Projects           | Daily scrum meeting                                                 | -                                            | -          | -          | -            |
| Friday<br>06/16/17   | Projects           | Daily scrum meeting                                                 | -                                            | -          | -          | -            |

--

### Week 11: Project
| Day                  | Main Topics        | During day                                                          | Repo                                         | Lead       |On Deck     | Slides       |
|:---------------------|:-------------------|:--------------------------------------------------------------------|:--------------------------------------------:|:-----------|:----------:|:------------:|
| Monday<br>06/19/17   | Projects           | Practice presentations 1 (CODE FREEZE)                              | -                                            | -          | -          | -            |
| Tuesday<br>06/20/17  | Projects           | Practice presentations 2                                            | -                                            | -          | -          | -            |
| Wednesday<br>06/21/17| Projects           | Dress rehearsal                                                     | -                                            | -          | -          | -            |
| Thursday<br>06/22/17 | Projects           | *Capstone showcase day*                                             | -                                            | -          | -          | -            |
| Friday<br>06/23/17   | *off*              | *relax day*                                                         | -                                            | -          | -          | -            |

--
### Week 12: Career services week
| Day                  | Main Topics              | During day                                                                       | Repo                       | Lead   |On Deck| Slides      |
|:---------------------|:-------------------------|:---------------------------------------------------------------------------------|:--------------------------:|:-------|:-----:|:-----------:|
| Monday<br>06/26/17   | Review, Whiteboarding    | CAREER SERVICES                                                                  | [interview-prep][137.2]    | -      | -     | -           |
| Tuesday<br>06/27/17  | Big O, Mock interviews   | CAREER SERVICES                                                                  | -                          | -      | -     | -           |
| Wednesday<br>06/28/17| Mock interviews          | CAREER SERVICES                                                                  | -                          | -      | -     | -           |
| Thursday<br>06/29/17 | Model comparison, Review | Salary negotiation (career services)                                             | -                          | -      | -     | -           |
| Friday<br>06/30/17   | -                        | Graduation                                                                       | -                          | -      | -     | -           |

--

# Weekly Assessements (my answers)

* [Assessment 1](https://github.com/kykiefer/assessment-day1)
* [Assessment 2](https://github.com/kykiefer/assessment-2)
* [Assessment 3]()
* [Assessment 4]()
* [Assessment 5]()
* [Assessment 6]()

<!-- Week 0 -->
[0.0]: https://github.com/zipfian/python-workshop
[0.01]: https://github.com/zipfian/python-workshop/tree/master/day1
[0.02]: https://github.com/zipfian/python-workshop/tree/master/day2
[0.03]: https://github.com/zipfian/python-workshop/tree/master/day3
[0.04]: https://github.com/zipfian/python-workshop/tree/master/day4
[0.1]: https://wiki.python.org/moin/BeginnersGuide
[0.2]: https://docs.python.org/2/tutorial/datastructures.html
[0.3]: https://www.cis.upenn.edu/~bcpierce/courses/629/papers/Java-tutorial/java/objects/index.html


<!-- Week 1 -->
[1]: notes/workflow.md
[2]: notes/pairing.md
[2.1]: https://github.com/zipfian/assessment-day1
[3]: https://github.com/zipfian/python-intro
[3.1]: https://github.com/zipfian/DSI_Lectures/tree/master/python-intro
[3.2]: http://learnpythonthehardway.org/book/ex40.html
[3.3]: https://github.com/zipfian/DSI_Lectures/tree/master/OOP
[8]: https://github.com/zipfian/OOP
[10]: http://sqlzoo.net/wiki/Main_Page
[11]: http://www.postgresql.org/docs/7.4/static/tutorial-start.html
[12]: http://blog.codinghorror.com/a-visual-explanation-of-sql-joins/
[13]: https://github.com/zipfian/sql
[13.1]: https://github.com/zipfian/DSI_Lectures/tree/master/sql
[18]: http://nbviewer.ipython.org/github/jvns/pandas-cookbook/blob/master/cookbook/A%20quick%20tour%20of%20IPython%20Notebook.ipynb
[19]: http://pandas.pydata.org/pandas-docs/stable/10min.html
[20]: http://nbviewer.ipython.org/github/cs109/content/blob/master/lec_04_wrangling.ipynb
[21]: http://manishamde.github.io/blog/2013/03/07/pandas-and-python-top-10/
[22]: http://nbviewer.ipython.org/github/cs109/content/blob/master/labs/lab3/lab3full.ipynb
[23]: https://github.com/zipfian/pandas
[23.1]: https://github.com/zipfian/DSI_Lectures/tree/master/sql-python
[23.2]: https://github.com/zipfian/DSI_Lectures/tree/master/pandas
[24]: https://github.com/zipfian/graphing-basics
[24.1]: https://github.com/zipfian/pandas-seaborn
[24.2]: https://github.com/zipfian/DSI_Lectures/tree/master/pandas-seaborn
[25.1]: https://github.com/zipfian/sql-python

<!-- Week 2 -->
[25]: http://cs229.stanford.edu/section/cs229-prob.pdf
[25.01]: https://oneau.wordpress.com/2011/02/28/simple-statistics-with-scipy/
[28.0]: https://github.com/zipfian/assessment-week2
[28.1]: https://github.com/zipfian/DSI_Lectures/tree/master/probability
[28]: https://github.com/zipfian/probability
[38]: https://github.com/zipfian/estimation-sampling
[38.0]: http://onlinestatbook.com/2/estimation/confidence.html
[38.01]: http://onlinestatbook.com/2/estimation/mean.html
[38.1]: https://www.youtube.com/watch?v=_nhgHjdLE-I
[38.2]: https://www.youtube.com/watch?v=I_dhPETvll8
[38.3]: https://github.com/zipfian/DSI_Lectures/tree/master/estimation-sampling
[39]: https://github.com/zipfian/ab-testing
[39.1]: https://www.youtube.com/watch?v=5ABpqVSx33I
[39.2]: https://www.youtube.com/watch?v=-FtlH4svqx4
[39.3]: https://github.com/zipfian/DSI_Lectures/tree/master/ab-testing
[39.4]: https://www.youtube.com/watch?v=lHI5oEgNkrk
[40]: https://github.com/zipfian/power-bayesian
[40.0]: https://www.youtube.com/watch?v=i567qvWejJA&index=15&list=PLFDbGp5YzjqXQ4oE4w9GVWdiokWB9gEpm
[40.1]: https://www.khanacademy.org/math/probability/statistics-inferential/sampling_distribution/v/central-limit-theorem
[40.2]: https://github.com/zipfian/DSI_Lectures/tree/master/power-bayesian
[40.3]: https://www.youtube.com/watch?v=r0tRgR74n_g&index=28&list=PLFDbGp5YzjqXQ4oE4w9GVWdiokWB9gEpm
[40.4]: http://stevehanov.ca/blog/index.php?id=132
[40.5]: https://github.com/zipfian/DSI_Lectures/tree/master/multi-armed-bandit
[44]: https://github.com/zipfian/multi-armed-bandit

<!-- Week 3 -->
[45]: https://github.com/zipfian/precourse/blob/master/Chapter_2_Linear_Algebra/notes.md
[45.1]: http://cs229.stanford.edu/section/cs229-linalg.pdf
[45.2]: https://github.com/zipfian/DSI_Lectures/tree/master/linear-algebra-eda
[45.3]: https://github.com/zipfian/DSI_Lectures/tree/master/linear-regression
[48]: https://github.com/zipfian/linear-algebra-eda
[51]: http://cs229.stanford.edu/notes/cs229-notes1.pdf
[52]: https://github.com/zipfian/gradient-descent
[54]: https://github.com/zipfian/regularized-regression
[54.1]: https://github.com/zipfian/DSI_Lectures/tree/master/regularized-regression
[54.2]: https://github.com/zipfian/DSI_Lectures/tree/master/logistic-regression
[54.3]: https://github.com/zipfian/DSI_Lectures/tree/master/gradient-descent
[58]: https://github.com/zipfian/linear-regression

<!-- Week 4 -->
[47.1]: http://www-bcf.usc.edu/~gareth/ISL/ISLR%20Sixth%20Printing.pdf
[47.2]: http://www.r2d3.us/visual-intro-to-machine-learning-part-1/
[65]: https://github.com/zipfian/non-parametric-learners
[65.1]: https://github.com/zipfian/DSI_Lectures/tree/master/non-parametric-learners
[68]: https://github.com/zipfian/random-forest
[68.1]: https://github.com/zipfian/DSI_Lectures/tree/master/random-forest
[71]: https://github.com/zipfian/svm
[71.2]: https://github.com/zipfian/DSI_Lectures/tree/master/svm
[71.3]: https://github.com/zipfian/DSI_Lectures/tree/master/boosting
[71.4]: https://github.com/zipfian/DSI_Lectures/tree/master/profit-curve
[71.5]: https://github.com/zipfian/regression-case-study
[71.6]: https://github.com/zipfian/DSI_Lectures/tree/master/neural-network
[71.7]: https://www.cs.ox.ac.uk/people/nando.defreitas/machinelearning/
[71.8]: https://github.com/loribeerman/galaxy_classification



<!-- Week 5 -->
[75]: https://github.com/zipfian/precourse/tree/master/Chapter_8_Web_Awareness
[76]: readings/web_scrape/scraping_tutorial.md
[76.1]: http://openmymind.net/mongodb.pdf
[77]: https://github.com/zipfian/web-scraping
[77.1]: https://github.com/zipfian/DSI_Lectures/tree/master/web-scraping
[84]: https://github.com/zipfian/nlp
[84.1]: https://github.com/zipfian/DSI_Lectures/tree/master/nlp
[200]: https://github.com/zipfian/ml-case-study

<!-- Week 6 -->
[NN]: https://github.com/zipfian/mlp_cnn_rnn
[103.1]: https://github.com/zipfian/DSI_Lectures/tree/master/topicmodeling
[103.2]: https://github.com/zipfian/DSI_Lectures/tree/master/image_featurization
[103.4]: https://github.com/zipfian/DSI_Lectures/tree/master/recommendation-systems
[103.5]: https://github.com/zipfian/DSI_Lectures/tree/master/recommender-case-study
[103.6]: https://github.com/zipfian/DSI_Lectures/tree/master/time-series
[104]: https://github.com/zipfian/clustering
[104.1]: https://github.com/zipfian/DSI_Lectures/tree/master/clustering
[107]: https://github.com/zipfian/dimensionality-reduction
[107.1]: https://github.com/zipfian/DSI_Lectures/tree/master/dimensionality-reduction
[118]: https://github.com/zipfian/recommendation-systems

<!-- Week 7 -->
[hp-python]: https://github.com/zipfian/high_performance_python
[parallel-intro]: http://sebastianraschka.com/Articles/2014_multiprocessing.html
[multiproc-python]:https://www.youtube.com/watch?v=X2mO1O5Nuwg
[threading]: http://pymotw.com/2/threading/
[spark_install]: https://github.com/zipfian/spark-install
[121.5]: https://github.com/zipfian/neural-networks/
[122]: https://github.com/zipfian/DSI_Lectures/tree/master/graphs
[122.2]: https://github.com/zipfian/DSI_Lectures/tree/master/high-performance-python
[122.3]: https://github.com/zipfian/DSI_Lectures/tree/master/map-reduce
[122.4]: https://github.com/zipfian/DSI_Lectures/tree/master/spark
[122.5]: https://github.com/zipfian/DSI_Lectures/tree/master/spark-aws
[129]: https://github.com/zipfian/data-at-scale
[131.1]: https://github.com/zipfian/final-assessment
[131.8]: https://aws.amazon.com/articles/4926593393724923

<!-- Week 8++ -->
[150.1]: https://github.com/zipfian/data-viz-for-ds
[150.2]: https://github.com/zipfian/DSI_Lectures/tree/master/data-viz-for-ds
[150.3]: http://www.w3schools.com/tags/ref_httpmethods.asp
[150.4]: https://github.com/zipfian/DSI_Lectures/tree/master/data-products
[150.5]: https://www.kaggle.com/wiki/DataScienceUseCases
[132.0]: https://github.com/zipfian/data-products
[132.1]: http://flask.pocoo.org/
[132.2]: http://blog.miguelgrinberg.com/post/the-flask-mega-tutorial-part-i-hello-world

[135]: https://github.com/zipfian/case-study
[137.2]: https://github.com/zipfian/interview-prep

[MLIA]: https://drive.google.com/file/d/0B1cm3fV8cnJwcUNWWnFaRWgwTDA/view?usp=sharing
[A3]: https://github.com/zipfian/assessment-3
[log-reg]: https://github.com/zipfian/logistic-regression
[boosting]: https://github.com/zipfian/boosting
[profit]: https://github.com/zipfian/profit-curve
[A4]: https://github.com/zipfian/assessment-4
[recursion]: http://interactivepython.org/runestone/static/pythonds/index.html#recursion
[DSBus]: https://drive.google.com/file/d/0B1cm3fV8cnJwNDJFNmx2a2RBaTg/view?usp=sharing
[proj]: https://github.com/zipfian/project-proposals
[esl]: http://web.stanford.edu/~hastie/local.ftp/Springer/OLD/ESLII_print4.pdf
[tfidf1]: http://blog.christianperone.com/?p=1589
[tfidf2]: http://blog.christianperone.com/?p=1747
[tfidf3]: http://blog.christianperone.com/?p=2497
[NLP]: http://victoria.lviv.ua/html/fl5/NaturalLanguageProcessingWithPython.pdf
[A5]: https://github.com/zipfian/assessment-5
[mmd9]: http://infolab.stanford.edu/~ullman/mmds/ch9.pdf
[mmd10]: http://infolab.stanford.edu/~ullman/mmds/ch10.pdf
[mmd11]: http://infolab.stanford.edu/~ullman/mmds/ch11.pdf
[nmf]: https://github.com/zipfian/topicmodeling
[rec2]: https://github.com/zipfian/alt-recommender-case-study
[nmf-reading]: http://www.quuxlabs.com/blog/2010/09/matrix-factorization-a-simple-tutorial-and-implementation-in-python/
[graphlab-rec]: https://turi.com/learn/userguide/recommender/choosing-a-model.html
[mrbook]: http://lintool.github.io/MapReduceAlgorithms/MapReduce-book-final.pdf
[time-series]: https://github.com/zipfian/time-series
[spark]: https://github.com/zipfian/spark
[spark-aws]: https://github.com/zipfian/spark-aws
[mftr]: http://www.columbia.edu/~jwp2128/Teaching/W4721/papers/ieeecomputer.pdf
[rec-paper]: http://files.grouplens.org/papers/algorithmstouserexperience.pdf
[funcprog]: https://docs.python.org/2/howto/functional.html#built-in-functions
[generators]: https://docs.python.org/2/howto/functional.html#generators
[mmd2]: http://infolab.stanford.edu/~ullman/mmds/ch2.pdf
[arima]: http://conference.scipy.org/proceedings/scipy2011/pdfs/statsmodels.pdf
[hyndman]: https://www.otexts.org/fpp
[timeseries]: readings/TimeSeries.pdf
[LearningSpark]: https://drive.google.com/file/d/0B1cm3fV8cnJwc2ZnMFJmT2RLOXM/view?usp=sharing
[sna]: readings/Social_Network_Analysis_for_Startups.pdf
[graphs]: http://github.com/zipfian/graphs
