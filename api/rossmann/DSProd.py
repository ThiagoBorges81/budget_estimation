import inflection
import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer
from sklearn.preprocessing import RobustScaler, MinMaxScaler, LabelEncoder, OneHotEncoder


class DSProd( object ):
    """
    Class for data preprocessing and feature engineering for a machine learning prediction project.
    """

    def __init__(self):
        """
        Initialize DSProd object.
        """
        pass

    def data_cleaning(self, df):
        """
        Clean the input DataFrame by formatting column names and removing rows with zero sales.

        Parameters:
        -----------
        df : pandas.DataFrame
            Input DataFrame to be cleaned.

        Returns:
        --------
        pandas.DataFrame
            Cleaned DataFrame.
        """

        # Format column names for convenience
        cols_old = ['Store'                     ,'DayOfWeek'                ,'Date'               ,
                    'Open'                      ,'Promo', 'StateHoliday'    ,'SchoolHoliday'      ,
                    'StoreType'                 ,'Assortment'               ,'CompetitionDistance',
                    'CompetitionOpenSinceMonth' ,'CompetitionOpenSinceYear' ,'Promo2'             ,
                    'Promo2SinceWeek'           ,'Promo2SinceYear'          ,'PromoInterval']

        snakecase = lambda x: inflection.underscore( x )

        cols_new = list( map( snakecase, cols_old ) )

        # rename
        df.columns = cols_new

        # Ensure that the date variable is in a proper format
        df['date'] = pd.to_datetime(df['date'])

        # # Filter out data with no sales registered
        # df = df.loc[df['sales'] != 0, :]

        # # Check if 'sales' column exists before filtering
        # if 'sales' in df.columns:
        #     # Filter out data with no sales registered
        #     df = df.loc[df['sales'] != 0, :]
        # else:
        #     print("Warning: 'sales' column not found. Skipping filtering.")
    
        return df


    def time_attributes(self, df):
        """
        Extracts time-based attributes from a DataFrame containing a 'date' column.

        Parameters:
        -----------
        df : pandas.DataFrame
            Input DataFrame containing a 'date' column from which time-based attributes
            will be extracted.

        Returns:
        --------
        pandas.DataFrame
            DataFrame with additional time-based attributes including 'year', 'month',
            'day', 'week_of_year', and 'year_week'.

        Notes:
        ------
        This function extracts various time-based attributes from a 'date' column in the
        input DataFrame, including year, month, day, week of the year, and year-week.
        The 'date' column is assumed to be in datetime format.

        Example:
        --------
        # Call the function to extract time-based attributes from the DataFrame df
        df_with_time_attrs = time_attr(df)
        """
        #Ensure th date variable is is proper format:
        df['date'] = pd.to_datetime(df['date'])

        # year
        df['year'] = df['date'].dt.year

        # month
        df['month'] = df['date'].dt.month

        # day
        df['day'] = df['date'].dt.day

        # week of year
        df['week_of_year'] = df['date'].dt.isocalendar().week

        return df


    def data_imputer(self, df):
        """
        Imputes missing values in specified columns of a DataFrame using KNNImputer.

        Parameters:
        -----------
        df : pandas.DataFrame
            Input DataFrame containing columns with missing values to be imputed.

        Returns:
        --------
        pandas.DataFrame
            DataFrame with missing values imputed using KNNImputer.

        Notes:
        ------
        This function scans the variables to check whether they have missing values from >0 to <10%. 
        Then, it replaces missing values in the specified columns of the input DataFrame
        using KNNImputer from the scikit-learn library. It imputes missing values based on
        the k-nearest neighbors of the data points with missing values. The number of neighbors
        used for imputation is set to 5 by default.

        Example:
        --------
        # Import required libraries
        from sklearn.impute import KNNImputer

        # Call the function to impute missing values in the DataFrame df
        df_imputed = data_imputer(df)
        """
        # Calculate the proportion of missing values for each column
        missing_proportions = df.isna().sum() / df.shape[0] * 100

        # Initialize an empty list to store column names with more than 10% missing values
        columns_with_missing = []

        # Iterate over each column's missing proportion
        for column, proportion in missing_proportions.items():
            if proportion > 10:
                # Drop column with moe than 10% missing
                df.drop(column, axis=1, inplace=True)
            elif 0 < proportion < 10:
                columns_with_missing.append(column)

        # Check if there are anyy columns with missing values
        if columns_with_missing:
            # Initialize KNNImputer
            imputer = KNNImputer(n_neighbors=5)

            # Validate input data for imputation
            if df[columns_with_missing].empty:
                print("Warning: Input data for imputation is empty.")

                return df

            # Impute missing values
            df[columns_with_missing] = imputer.fit_transform(df[columns_with_missing])
        else:
            print("No columns with missing found. Skipping imputation.")
            
        return df

    def categorical_format(self, df):
        """
        Formats categorical variables in the given DataFrame.

        Args:
            df (pandas.DataFrame): The input DataFrame containing categorical variables.

        Returns:
            pandas.DataFrame: The DataFrame with formatted categorical variables.

        Example:
            df = cat_format(df)

        """
        # Fix data type
        df['competition_distance'] = df['competition_distance'].astype(int)

        # # assortment
        df['assortment'] = df['assortment'].apply( lambda x: 'basic' if x == 'a' else 'extra' if x == 'b' else 'extended' )

        # # state holiday
        df['state_holiday'] = df['state_holiday'].apply( lambda x: 'public_holiday' if x == 'a' else 'easter_holiday' if x == 'b' else 'christmas' if x == 'c' else 'regular_day' )

        return df

    def rescaling_vars(self, df):
        """
        Rescales numerical variables in the given DataFrame using RobustScaler and MinMaxScaler.

        Args:
            df (pandas.DataFrame): The input DataFrame containing numerical variables.

        Returns:
            pandas.DataFrame: The DataFrame with rescaled numerical variables.

        Example:
            df = rescaling_vars(df)
        """
        mms = MinMaxScaler()

        # # year
        df['year'] = mms.fit_transform( df[['year']])

        return df

    def encode_vars(self, df):
        """
        Encodes categorical variables in the given DataFrame using different techniques.

        Args:
            df (pandas.DataFrame): The input DataFrame containing categorical variables.

        Returns:
            pandas.DataFrame: The DataFrame with encoded variables.

        Example:
            df = encode_vars(df)
        """

        le = LabelEncoder()

        # state_holiday - One Hot Encoding
        # df = pd.get_dummies( df, prefix=['state_holiday'], columns = ['state_holiday'] )

        # store_type - Label Encoding
        df['store_type'] = le.fit_transform( df['store_type'] )

        # assortment - Ordinal Encoding
        df['assortment_encoded'] = le.fit_transform(df['assortment'])

        return df

    def log_transform_vars(self, df):
        """
        Applies a natural logarithm transformation (log1p) to the 'sales' column in the given DataFrame.

        Args:
            df (pandas.DataFrame): The input DataFrame containing the 'sales' column.

        Returns:
            pandas.DataFrame: The DataFrame with the transformed 'sales' column.

        Example:
            df = log_trf_var(df)
        """
        if 'sales' in df.columns:
            df['sales'] = np.log1p(df['sales'])
        else:
            pass

        return df

    def nature_transform_vars(self, df):
        """
        Apply trigonometric transformations to date-related columns in a DataFrame.

        Args:
            df (pandas.DataFrame): Input DataFrame containing columns 'day_of_week',
                                  'month', 'day', and 'week_of_year'.

        Returns:
            pandas.DataFrame: Transformed DataFrame with additional columns:
                - 'day_of_week_sin' and 'day_of_week_cos': Sine and cosine of day of week.
                - 'month_sin' and 'month_cos': Sine and cosine of month.
                - 'day_sin' and 'day_cos': Sine and cosine of day.
                - 'week_of_year_sin' and 'week_of_year_cos': Sine and cosine of week of year.
        """
        # day of week
        df['day_of_week_sin'] = df['day_of_week'].apply( lambda x: np.sin( x * ( 2. * np.pi/7 ) ) )
        df['day_of_week_cos'] = df['day_of_week'].apply( lambda x: np.cos( x * ( 2. * np.pi/7 ) ) )

        # month
        df['month_sin'] = df['month'].apply( lambda x: np.sin( x * ( 2. * np.pi/12 ) ) )
        df['month_cos'] = df['month'].apply( lambda x: np.cos( x * ( 2. * np.pi/12 ) ) )

        # day 
        df['day_sin'] = df['day'].apply( lambda x: np.sin( x * ( 2. * np.pi/30 ) ) )
        df['day_cos'] = df['day'].apply( lambda x: np.cos( x * ( 2. * np.pi/30 ) ) )

        # week of year
        def weeks_in_year(year):
            if year % 4 == 0 and (year % 100 != 0 or year % 400 == 0):

                return 53  # Leap year
            else:
                return 52  # Non-leap year

        df['week_of_year_sin'] = df['week_of_year'].apply( lambda x: np.sin( x * ( 2. * np.pi/weeks_in_year(x) ) ) )
        df['week_of_year_cos'] = df['week_of_year'].apply( lambda x: np.cos( x * ( 2. * np.pi/weeks_in_year(x) ) ) )

        return df

    def one_hot_encoder(self, df, cat_vars=['state_holiday','assortment','store_type']):
        """
        One-hot encodes categorical variables in the given DataFrame.

        Args:
            df (pandas.DataFrame): The input DataFrame containing categorical variables.
            cat_vars (list): A list of categorical variable names to be one-hot encoded.

        Returns:
            pandas.DataFrame: The DataFrame with encoded variables.

        Example:
            df = one_hot_encoder(df, cat_vars=['state_holiday','assortment','store_type'])
        """
        # Check whether all the variables are present in the dataset
        if any(var in df.columns for var in cat_vars):
            ohe = OneHotEncoder(sparse_output=False).set_output(transform='pandas')

            # Filter out only the variables present in the data frame
            cat_vars_present = [var for var in cat_vars if var in df.columns]

            #Perform one hot encoding on the present categorical variables
            ohe_data = ohe.fit_transform(df[cat_vars_present])

            # Create DataFrame with the one-hot encoded features and column names
            encoded_df = pd.DataFrame(ohe_data, columns=ohe.get_feature_names_out(cat_vars), index=df.index)

            # Concatenate the encoded DataFrame with the original DataFrame
            df_encoded = pd.concat([df, encoded_df], axis=1)
            df_encoded = df_encoded.drop(cat_vars, axis=1)
            df = df_encoded

            return df_encoded

        else:
            print("Warning: None of he specified variables found")

        cols_selected = [   'store',
                            'customers',
                            'promo',
                            'competition_distance',
                            'promo2',
                            'assortment_basic',
                            'assortment_extra',
                            'store_type_b',
                            'store_type_d']
        

        return df[cols_selected]
    
    def get_prediction( self, model, df, test_data ):
        """
        Generate predictions using a trained model and input DataFrame.

        Parameters:
        -----------
        model : object
            Trained machine learning model for prediction.
        df : pandas.DataFrame
            Input DataFrame containing features.
        test_data : array-like
            Test data for prediction.

        Returns:
        --------
        str
            JSON representation of the DataFrame with predictions.
        """
        df = df.drop(['date','promo_interval'], axis=1)
        test_data = test_data.drop(['date','promo_interval'], axis=1)

        # prediction
        pred = model.predict( test_data )

        # join pred into the original data
        df['prediction'] = np.expm1( pred )

        return df.to_json( orient='records', date_format='iso' )
