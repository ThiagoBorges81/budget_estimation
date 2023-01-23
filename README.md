# Data Science to Production

_Disclaimer:_ All the scenarios, descriptions, and situations in this project are fictional and do not represent any of this company’s views, actions, or opinions. The processes, results, recommendations, and any other information provided in this project have the sole intention to describe my Data Science skills with the approach and solutions provided. The data used in this project can be retrieved from a public source at (https://www.kaggle.com/competitions/rossmann-store-sales/overview). The data had been used in Kaggle public competition and complies with Kaggle's terms and conditions.  

**Introduction**  

Hello and welcome! This is an end-to-end Data Science project aimed to develop an idea and practice my Data Science skills. This project is one of a series where I will answer business questions using Data Science techniques. This one in particular aims to solve a fictional business question using a Regression technique approach. I will also use productive business methodologies aiming to bring value and revenue to a (fictional) business. Even though this is a practicing project, all techniques used here are real-world solutions that can be implemented in any business in need to predict and estimate future revenue. Well without further ado, let’s get started!  

**1. Business problem**  
    
**The context for the project**  

Rossmann is a large company and operates over 3000 chemist stores in Europe. As part of the company’s operations, several store managers ensure all the processes run as planned. The CFO of the company needs to plan a budget for the refurbishment of the stores and in one of the monthly meetings with the store managers, the CFO brought a request for them to provide an estimate of the sales for the next six weeks so that the budget for the refurbishment could be allocated, based on this estimation. Therefore, the store managers were tasked to provide the CFO with a sale estimate for the next six weeks.

**_Business question_**

Rossmann’s CFO needs to know what are the sales prospects for the next six weeks.

**2. Planning the solution**  

To address the CFO’s requirements, there are few seps that need to be accomplished being by the data scientist which include establishing the problem to be solved, understanding the business and mapping the assumptions, collecting, processing and analysing the data, implementing the machine learning models and assessing the results, deploying the solution and providing a simple and effective tool where the stakeholders could use the final product requested.  

**3. Assumptions**  

After a careful studying of the CFO’s request, a few assumption of the business were made, in order to better understand a number of aspects of the business to deliver the best final product. Below is a description of few characteristics that may impact on store performance. Features that may impact on daily store sales include stores’ location, clients’ characteristics, products availability, the store’s characteristics, the time of the year, to mention the main ones. The mind map would allow me to understand the phenomena of what I am modeling, and what are my agents of interest as well as better define my hypothesis. These can be validated or refused during the exploratory data analysis.  

![alt text]('/media/thiago/THIAGO1TB/Thiago/2022/data_science/seja_um_data_scientist/CDS/ds_em_producao/ds_producao_projeto/img/mind_map_EN.png')  

**4. Solution development**  


**Data collection**  

The data for this project was collected from a Kaggle competition. According to Kaggle’s website, the dataset represents historical sales data for 1115 Rossmann stores. Data are available in spreadsheets and I have downloaded the data directly to my hard drive for processing, analysis, and modeling. Data can be downloaded [here:](https://www.kaggle.com/competitions/rossmann-store-sales/overview)  

The Kaggle web page stores the data used in this project in separate links. The data in .csv files are formatted as tables and available for download. Assuming the data come from the Rossmann database, any bias that exists may come from the registering system. The data collected is under Kaggle’s license agreement. This agreement ensures the client’s personally identifiable information remains anonymous. The data was stored in a physical drive (external hard drive) protected by a password. After downloading, I created a backup copy and stored the data. I inspected the data visually and prepared it for sorting, filtering, and cleaning. Also, ensuring data integrity would enhance the quality of the interpretation of the outcomes of the data analysis.  


**Data cleaning**  

Data cleaning comprises a few comprehensive steps to ensure data quality when we reach the modeling phase. These are described below:  

 1. Descriptive analysis  
 
In this step, a few actions are performed including verifying the amount of data available, what are the variable types, checking for missing data, and performing a descriptive statistic analysis.  

  1. First, to increase clarity and help during the modeling a few steps ahead, the names of the columns were formatted, from Camel case to snake case;  
  
  2. Then, after checking the data types, missing values were scanned. There were missing data in six variables of the dataset. The variables related to the date were treated using the actual selling date as a reference;  
  
  3. Below is the table for the descriptive statistics of the numerical features:  
  
These few simple steps are important to increase clarity and and quality of the data.  

  4. The categorical variables were also checked using visualizations.  
  

**Feature engineering**  

During the process of data cleaning, it is important to prepare the data so that I can have them ready for studying, particularly during the exploratory data analysis phase. The feature engineering phase also contributes to planning the solutions and the mind map comes in handy.  


**Variable filtering**  

Variable filtering is also part of data cleaning. Some constraints that the business presents may make it difficult to deploy the model into production. Therefore, filtering the variables not only will shape the data according to the business question but also will help choose the most relevant variables to the modeling.


**Data exploration**  

**_Exploratory data analysis (EDA)_**  

After a thorough process of data cleaning which used specific steps to first prepare the data, it is time to check how and the strength of the impact of these variables have on the phenomena I am investigating. In this phase, it is possible to detect nuances in the data that may influence the machine learning modeling in the next steps. Such an approach substantially decreases trial and error. Here, the data scientist will expand the business experience and will be able to generate insights, either bringing new information to the table or contrasting something that was believed to be true. Finally, EDA may surface important variables in the model.  

  • Univariate analysis  
  
The Univariate analysis would allow for a high level exploration of the data, where a few visualizations would lay groundwork for a more in depth analysis.


First, let’s investigate the distribution of the numerical variables:  



Then, let’s explore the distribution of the categorical variables:  



  • Bivariate analysis  


While the univariate analysis allows for a more visual inspection of the data, the bivariate analysis starts providing further insights from the data. As such, some hypotheses may be raised and tested. These are:



**H1 - Stores with more assortment should sell more.**  
_FALSE:_ The data show that stores with more assortment sell less (Figure H1-A). Also, there is a seasonal trend for the assortment for the Basic and Extended (Figure H1-B) and Extra (Figure H1 - C). See figure below:  

Figure H1 - A. Total sales comparison between different assortment.
          
Figure H1 - B. Total sales comparison between assortment over time.
          
Figure H1 - C. Total sales of extra assortment over time.
          
**H2 - Stores with close competition should sell less.**
_FALSE:_ The data show that stores with close competition sell more.
          

**H3 - Stores with longer competition should sell more.**
_FALSE:_ The data show that stores with longer competition sell less.


**H4 - Stores with longer sales should sell more.**
_FALSE:_ Data show that stores with longer sales sell less, after a period of the sale.


**H5 - Stores with consecutive sales should sell more.**  
_FALSE:_ Stores with consecutive sales sell less.


**H6 - Stores opened during the Christmas holiday should sell more.**
_FALSE:_ Data show that stores opened during the Christmas holiday sell less.  


**H7 - Stores should sell more over the years.**
_FALSE:_ Data show that stores sell less over the years.


**H8 - Stores should sell more during the second semester of the year.**  
_FALSE:_ Data show that stores sell less during the second semester of the year.  


**H9 - Stores should sell more after the 10th day of the month.**
_TRUE:_ Data show that stores do sell more after the 10th day of the month.  


**H10 - Stores should sell less over the weekends.**  
_TRUE:_ Data show that stores sell less during weekends.  


**H11 - Stores should sell less during school holidays.**  
_TRUE:_ Data show that stores sell less during school holidays, except over August.  



School holiday - 0=No; 1=Yes  



**• Multivariate analysis**  

The multivariate analysis is yet another important component of the project. Here, it is possible to see how the features correlate to each other. This would raise information on whether there is collinearity in the data as well as the strength of the relationships between the variables. See below a correlation matrix for:  

* Numerical attributes:

Here, the correlation method used is Pearson’s correlation.  

* Categorical attributes:  

For the categorical attributes, the method used was Cramer’s V correlation.



**Data preparation**  

Data preparation involves making the data meet certain requirements to optimize and increase the accuracy of machine learning. Machine Learning models are designed expecting the data input to be in a particular format. The three main data preparation process includes:  

• Normalization. 
  ◦ Normalization refers to transforming the mean to zero and the standard deviation to one;  
  
• Rescaling 
  ◦ To rescale the data to the interval between 0 and 1, to a normal (Gaussian) distribution;  
  
• Transformation 
  ◦ To transform categorical to numerical features (using encoding) 
  ◦ Transformation of nature 

**Data modeling**  

**_Feature selection_**  

Feature selection involves choosing the best features of the data to be modeled. Also, the fewer features the better as simpler models should be prioritized. In this project, I have used the Boruta algorithm to select the most important features of the data.  


**Algorithm’s performance**  

**_Machine Learning Algorithms_**  

After thorough processing of the data, it is time to test different Machine Learning models in the response variable, estimating the revenue six weeks ahead in the future for each store. As similar machine learning models may present distinct responses in the prediction, I have implemented simpler models as a baseline for comparison followed by more complex models. The models were:  

• 1. Average model: This model was the simplest in the project and served as a baseline for comparison with other models. The average model output the following results;
    
• 2. Linear Regression Model
      After the scores for baseline had been determined, I performed a linear regression with cross-validation that resulted in the following:  
      
• 3. Linear Regression Regularized Model - Lasso
      Implementing a regularized linear regression with cross-validation resulted in a slight change in performance:
      
• 4. Random Forest Regressor
      Another attempt to improve model performance was to implement a random forest model:
      
• 5. XGBoost Regressor
      Finally, an XGBoost regressor model was implemented to optimize sales prediction:  
      
Even though this model has not presented promising performance metrics, it tends to respond to hyperparameter fine tunning. Therefore, as a first attempt, I have chosen to optimize the model, before I would make a decision to discard it.  
      
**Algorithm’s Assessment**  


**_Hyperparameter Fine Tuning_**  

After fine tuning the model, I have reached the following performance metrics:  

The tuning returned a satisfactory result and I decided to use it in production.  


**Translating and interpreting the error**  

After performing all the data processing, and developing different machine learning models, I have decided to use the XGBoost due to its ability to respond well to fine-tuning. Then, I have calculated the predictions for the store, as well as estimated best and worst case scenarios for the revenue and provided model performance for that particular store:  


**Machine Learning Performance**  

In addition to translating the error, it was also possible to track the machine learning performance, as displayed in the figures below:  


In addition to store-by-store results, the algorithm also presented an overall performance:  


The predictions above show the gross income expected for the period and the worst and the best scenario estimation for revenue generation were included. Such information would help in planning the financial needs of the company.

**Deploying Model to Production**  

At this stage of the project, the product is published on the internet and it is how my client/ Boss will use the product. The model can be accessed using the mobile app Telegram. In this case, a bot was created and the user can check the results they need right from their mobile phones. You can try using the bot yourself by either clicking on the link or pointing you mobile phone camera to the QRCode and access the bot:  



Once you are on the bot’s page, type /+ a number between 0 and 1000 to check the revenue estimate for the next six weeks for that store number you typed. For example, if you type /22, you will receive a message saying: “Store Number 22 will sell R$171,358.32 in the next 6 weeks.” This is valid for all teh store within the database. If you type words or numbers that are not in the database, you get an error message.



       
**Final considerations on the project**  


This is an end-to-end data science solution, where a business question was addressed using data. Moreover, several data science techniques were employed to deliver an elegant and useful solution. The main stakeholder may use the solution on-the-go to make important decisions and increase the financial efficiency of the company.


**Next steps**  

There are few actions that can be made to improve the precision of the machine learning model including testing different models and improving the modeling fine tuning. Also, including updated data in the analysis would provide a current scenario of the stores. Finally, there may be other business questions that need addressing so that a new plan and execution is required.  

https://t.me/rossmann_cds_bot
