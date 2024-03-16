import pickle
import pandas as pd
from flask import Flask, request, Response
from DSProd import DSProd

model = pickle.load(open('/media/thiago/thiago_1tb/thiago/2022/data_science/seja_um_data_scientist/CDS/data_formation/ds_em_producao/ds_producao_projeto/model/model_rossmann.pkl','rb'))

app = Flask(__name__)

@app.route('/api/rossmann/predict', methods = ['POST'])

def rossmann_predict():
    test_json = request.get_json()

    if test_json:
        if isinstance( test_json, dict ):
            test_raw = pd.DataFrame (test_json, index = [0])
        
        else:
            test_raw = pd.DataFrame (test_json, columns = test_json[0].keys())

        # Instantiate DS Prod class
        pipeline = DSProd()

        # Data Cleaning
        df1  = pipeline.data_cleaning( test_raw )

        # Engineer time attributes
        df2  = pipeline.time_attributes ( df1 )

        # Impute missing data
        df3  = pipeline.data_imputer ( df2 )

        # Format categorical variables
        df4  = pipeline.categorical_format ( df3 )

        # Rescale selected variables
        df5  = pipeline.rescaling_vars ( df4 )

        # Encode selected variables
        df6  = pipeline.encode_vars ( df5 )

        # Log-transform the target variable 
        df7  = pipeline.log_transform_vars ( df6 )

        # Nature transform the cyclical variables
        df8  = pipeline.nature_transform_vars ( df7 )

        # One hot encode selected variables
        df9  = pipeline.one_hot_encoder ( df8 )

        # Prediction
        df_response = pipeline.get_prediction ( model, test_raw, df9 )

        return df_response.to_json(orient = 'records', date_format='iso'), 200, {'Content-Type':'application/json'}
    
    else:
        return Response( '{}'.format(test_json), status = 200, mimetype = 'application/json')
    
if  __name__ == '__main__':
    app.run( '0.0.0.0', debug = False)