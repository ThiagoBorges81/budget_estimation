# Data Science in Production

_Disclaimer:_ All the scenarios, descriptions, and situations in this project are fictional and do not represent any of this company’s views, actions, or opinions. The processes, results, recommendations, and any other information provided in this project have the sole intention to describe my Data Science skills with the approach and solutions provided. The data used in this project was retrieved from a public source [here](https://www.kaggle.com/competitions/rossmann-store-sales/overview). The data had been used in Kaggle public competition and complies with Kaggle's terms and conditions.
  

**Introduction**  

Hello and welcome! This is an end-to-end Data Science project aimed to develop and deliver a solution, and to practice my Data Science skills. This **Regression** project is one of a series where I will answer business questions using Data Science techniques. I will also use productive business methodologies aiming to bring value and revenue to a (fictional) business. Even though this is a practicing project, all techniques used here are real-world actionable solutions that can be implemented in any business in need to predict and estimate future revenue.
The approach that will guide the entire project is the CRISP-DS, taught in the Data Science Community, where I am a member and have been studying and developing my Data Science skills (Figure 1).

![alt text](/img/crisp_ds.png "CRISP-DS")
Figure 1. CRISP-DS

The CRISP-DS is inspired by the CRISP-DM, which stands for Cross Industry Standard Process for Data Mining. This methodology breaks down the project into smaller tasks aiming to make the process of delivering solutions faster as well as to ease problem detection throughout the cycle. These processes are well-defined steps and they allow the business team to be on the same page throughout the development of the project.  

Well, without further ado, let’s get started!

**1. Business problem**  
    
**1.1. The context for the project**  

Rossmann is a large company and operates over 3000 chemist stores in Europe. As part of the company’s operational startegy, several store managers ensure all the processes run as planned. The CFO of the company needs to plan a budget for the refurbishment of the stores and in one of the monthly meetings with the store managers, the CFO brought a request for them to provide an estimate of the sales for the next six weeks so that the budget for the refurbishment could be allocated, based on this estimation. Therefore, the store managers were tasked to provide the CFO with a sale estimate for the next six weeks.

**_1.2. Business question_**

Rossmann’s CFO needs to know what are the sales prospects for the next six weeks for each store.

**2. Planning the solution**  

To address the CFO’s requirements, there are few seps that need to be accomplished by the data scientist which include:
  * Establishing the problem to be solved;  
  * Understanding the business and mapping the assumptions;  
  * Collecting, processing and analysing the data;  
  * Implementing Machine Learning models and assessing the results;  
  * Deploying the solution;  
  * Providing a simple and effective tool where the stakeholders could use the final product requested.  

**3. Assumptions**  

After a careful studying of the CFO’s request, a few assumption of the business were made, in order to better understand a number of aspects of the business to deliver the best final product. Below is a description of few characteristics that may impact on store performance. Features that may impact on daily store sales include stores’ location, clients’ characteristics, products availability, the store’s characteristics, the time of the year, to mention the main ones. The mind map would allow me to understand the phenomena I am modeling (sales), and what are my agents of interest as well as better define my hypothesis. These can be validated or refused during the exploratory data analysis stage.  

![alt text](/img/mind_map_EN.png )  

Figure 2. Mind Map.  


**4. Solution development**  


**4.1. Data collection**  

The data for this project was collected from a Kaggle competition. According to Kaggle’s website, the dataset represents historical sales data for 1115 Rossmann stores. Data are available in spreadsheets and I have downloaded the data directly to my hard drive for processing, analysis, and modeling. Data can be downloaded [here:](https://www.kaggle.com/competitions/rossmann-store-sales/overview)  

The Kaggle web page stores the data used in this project in separate links. The data in .csv files, formatted as tables are available for download. Assuming the data come from the Rossmann database, any bias that exists may come from the registering system.  

The data collected is under Kaggle’s license agreement. This agreement ensures the client’s personally identifiable information remains anonymous. The data was stored in a physical drive (external hard drive) protected by a password. After downloading, I created a backup copy and stored the data. I inspected the data visually and prepared it for sorting, filtering, and cleaning. Also, ensuring data integrity would enhance the quality of the interpretation of the outcomes of the data analysis.  


**4.1.2. Data cleaning**  

Data cleaning comprises a few comprehensive steps to ensure data quality when we reach the modeling phase. These are described below:  

 **4.1.2.1. Descriptive analysis**
 
In this step, a few actions were performed including verifying the amount of data available, what are the variable types, checking for missing data, and performing a descriptive statistic analysis.

  1. First, to increase clarity and help during the modeling a few steps ahead, the names of the columns were formatted, from Camel case to snake case;  
  
  2. Then, after checking the data types, missing values were scanned. There were missing data in six variables of the dataset. The variables related to the date were treated using the actual selling date as a reference;  
  
  3. Below is the table for the descriptive statistics of the numerical features:  

  Table 1. Descriptive statistics of numerical variables

  ![alt text](/img/desc_stats_numeric.png )  
  
These few simple steps are important to increase clarity and and quality of the data.  

  4. The categorical variables were also checked using visualizations.  
   ![alt text](/img/cat_attributes.png )  
Figure 3. Boxplots of the categorical variables. 
  

**4.1.2.2. Feature engineering**  

During the process of data cleaning, it is important to prepare the data so that I can have them ready for studying, particularly during the exploratory data analysis phase. The feature engineering phase also contributes to planning the solutions and the mind map comes in handy.  


**4.1.2.3. Variable filtering**  

Variable filtering is also part of data cleaning. Some constraints that the business presents may make it difficult to deploy the model into production. Therefore, filtering the variables not only will shape the data according to the business question but also will help choose the most relevant variables to the modeling.


**4.2. Data exploration**  

**_4.2.1. Exploratory data analysis (EDA)_**  

After a thorough process of data cleaning which used specific steps to first prepare the data, it is time to check how and the strength of the impact of these variables have on the phenomena I am investigating. In this phase, it is possible to detect nuances in the data that may influence the machine learning modeling in the next steps. Such an approach substantially decreases trial and error. Here, I would expand the business experience and will be able to generate insights, either bringing new information to the table or contrasting something that was believed to be true. Finally, EDA may surface important variables in the model.  

* **Univariate analysis**
  
  The Univariate analysis would allow for a high level exploration of the data, where a few visualizations would lay groundwork for a more in depth analysis.


  First, let’s investigate the distribution of the numerical variables:  

  ![alt text](/img/hist_num_variables.png )
  Figure 4. Histogram of the numerical variables.



Then, let’s explore the distribution of the categorical variables:  

  ![alt text](/img/dist_cat_variables.png )  
  Figure 5. Distribution of the categorical variables.



  • Bivariate analysis  


While the univariate analysis allows for a more visual inspection of the data, the bivariate analysis starts providing further insights from the data. As such, some hypotheses may be raised and tested. These are:



**H1 - Stores with more assortment should sell more.**  
_TRUE:_ The data show that stores with more assortment sell less (Figure H1-A). Also, there is a seasonal trend for the assortment for the Basic, Extended, and Extra (Figure H1-B). See figure below:  

![alt text](/img/h1a.png )
Figure H1 - A. Comparison of the sales average based on store assortment.  


![alt text](/img/h1b.png )
Figure H1 - B. Total sales comparison between assortment over time.

          
**H2 - Stores with close competition should sell less.**  
_FALSE:_ The data show that stores with close competition sell more.  

![alt text](/img/h2.png )
Figure H2. Sales based on competition proximity.

**H3 - Stores with longer competition should sell more.**  
_FALSE:_ The data show that stores with longer competition sell less.  

![alt text](/img/h3.png )
Figure H3. Sales based on competition duration.


**H9 - Stores should sell more after the 10th day of the month.**  
_TRUE:_ Data show that stores do sell more after the 10th day of the month.  

![alt text](/img/h9.png )  
Figure H9. Sales based on day of the month.

**H10 - Stores should sell less over the weekends.**  
_TRUE:_ Data show that stores sell less during weekends. 

![alt text](/img/h10.png )
Figure H10. Sales based on days of the week.


**H11 - Stores should sell less during school holidays.**  
_TRUE:_ Data show that stores sell less during school holidays, except during August.  

![alt text](/img/h13.png )
Figure H11. Sales based on school calendar.


School holiday - 0=No; 1=Yes  



**• Multivariate analysis**  

The multivariate analysis is yet another important component of the project. Here, it is possible to see how the features correlate to each other. This would raise information on whether there is collinearity in the data as well as the strength of the relationships between the variables. See below a correlation matrix for:  

* Numerical attributes:  

![alt text](/img/multivarHeatMap.png)  
Figure 6. Heatmap of the correlation matrix of the numerical variables.

Here, the correlation method used is Pearson’s correlation.  

* Categorical attributes:  

![alt text](/img/CramersVHeatMap.png)
Figure 7. Heatmap of the correlation matrix of the categorical variables.  

For the categorical attributes, the method used was Cramer’s V correlation.



**4.3. Data preparation**  

Data preparation involves preparing the data meet certain requirements to optimize and increase the accuracy of machine learning. Machine Learning models are designed expecting the data input to be in a particular format. The three main data preparation process includes:   

* Normalization. 
  * Normalization refers to transforming the mean to zero and the standard deviation to one;  

* Rescaling  
  * To rescale the data to the interval between 0 and 1, to a normal (Gaussian) distribution;  

* Transformation  
  * To transform categorical to numerical features (using encoding)
  * Transformation of nature - Based on the characteristics of the data.

**4.4. Data modeling**  

**_4.4.1. Feature selection_**  

Feature selection involves choosing the best features of the data to be modeled. Also, the fewer features the better as simpler models should be prioritized. In this project, I have used the Boruta algorithm to select the most important features of the data.  


**5. Algorithm’s performance**  

**_5.1. Machine Learning Algorithms_**  

After thorough processing of the data, it is time to test different Machine Learning models in the response variable, estimating the revenue six weeks ahead in the future for each store. As similar machine learning models may present distinct responses in the prediction, I have implemented simpler models as a baseline for comparison followed by more complex models. The models were:  

* 1. Average model: This model was the simplest in the project and served as a baseline for comparison with other models;  
    
* 2. Linear Regression Model: After the scores for baseline had been determined, I performed a linear regression with cross-validation;    
      
* 3. Linear Regression Regularized Model - Lasso: Implementing a regularized linear regression with cross-validation;  
      
* 4. Random Forest Regressor: Another attempt to improve model performance was to implement a random forest model;  
      
* 5. XGBoost Regressor: Finally, an XGBoost regressor model was implemented to optimize sales prediction.  
      
The chosen model to be sent to production was the XGBoost. Even though this model has not presented best performance metrics (it's ranked 2nd), the size of the model was significanlty lower as well as the time to process was also smaller. Therefore, I had decided to use this algorithm as a trade of between time and size, for optimizing resources.    
      
**5.2. Algorithm’s Assessment**  


**_5.2.1. Hyperparameter Fine Tuning_**  

After fine tuning the model, I have reached the following performance metrics:  

| Model Name | MAE | MAPE | RMSE |
| --- | --- | --- | --- |
| XGBoost Regressor | 760.05 | 0.11 | 1088.44 |
|


The tuning returned a satisfactory result and I decided to use it in production.  


**6. Translating and interpreting the error**  

After performing all the data processing, and developing different machine learning models, I have decided to use the XGBoost due to the cost effective results. Then, I have calculated the predictions for the stores, as well as estimated best and worst case scenarios for the revenue and provided model performance for that particular store:  

| Store | Predictions | Worst Scenario | Best Scenario | MAE | MAPE |
| --- | --- | --- | --- | --- | --- | 
| 292 | 104033.08 | 100714.97 | 107351.18 | 3318.10 | 0.57 |
| 909 | 238233.88 | 230573.34 | 245894.41 | 7660.54 | 0.52 |
| 976 | 203030.16 | 199110.95 | 206949.36 | 3919.20 | 0.31 |
| 722 | 353005.78 | 351013.63 | 354997.94 | 1992.16 | 0.27 |
| 595 | 400883.63 | 397415.26 | 404351.99 | 3468.36 | 0.24 |
|


**7. Machine Learning Performance**  

In addition to translating the error, it was also possible to track the machine learning performance, as displayed in the figures below:  

![alt text](/img/ml_performance.png)  
Figure 8. Machine learning performance.

In addition to store-by-store results, the algorithm also presented an overall performance:  

| Scenario | Values |
| --- | --- |
| Predictions | $ 287,260,416.00 |
| Worst Scenario | $ 286,409,667.70 |
| Best Scenario | $ 288,111,145.69 |
|


The predictions above show the gross income expected for the period and the worst and the best scenario estimation for revenue generation were included (based on model references). Such information would help in planning the financial needs of the company.

**8. Deploying Model to Production**  

At this stage of the project, the product is published on the internet and it is how my client/ Boss will use the product. The model can be accessed using the mobile app Telegram. In this case, a bot was created and the user can check the results they need right from their mobile phones. You can try using the bot yourself by either clicking on the link or pointing you mobile phone camera to the QRCode and access the [bot:](https://t.me/rossmann_cds_bot)  

![alt text](/img/qrcode_telg_bot.jpeg)
Figure 9. QRCode to access the Telegram Bot for sales prediction.



Once you are on the bot’s page, type /+ a number between 0 and 1000 to check the revenue estimate for the next six weeks for that store number you typed. For example, if you type /22, you will receive a message saying: “Store Number 22 will sell R$171,358.32 in the next 6 weeks.” This is valid for all teh store within the database. If you type words or numbers that are not in the database, you get an error message.



       
**9. Final considerations on the project**  


This is an end-to-end data science solution, where a business question was addressed using data. Moreover, several data science techniques were employed to deliver an elegant and useful solution. The main stakeholder may use the solution on-the-go to make important decisions and increase the financial efficiency of the company.


**10 . Next steps**  

There are few actions that can be made to improve the performance of the machine learning model including improving the feature engineering, testing different machine learning models and improving the modeling fine tuning. Also, including updated data in the analysis would provide a current scenario of the stores. Finally, there may be other business questions that need addressing so that a new plan and execution is required.