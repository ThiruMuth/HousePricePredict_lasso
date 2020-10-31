from flask import Flask, request, render_template
import joblib
import numpy as np
import pandas as pd

app = Flask(__name__)
model = joblib.load('house_price_lasso_model.pkl')

@app.route("/")
def hello():
    print("initial HTML call to render index html") 
    return render_template('index.html')

@app.route('/predict', methods = ['GET','POST'])
def result():
    if request.method == 'GET':
         print('GET') 
    if request.method == 'POST':
        print('POST')
        Postedby=request.form["Postedby"]
        if Postedby=="Dealer":
            POSTED_BY_DEALER=1
            POSTED_BY_BUILDER=0
            POSTED_BY_OWNER=0
        elif Postedby=="Builder":
            POSTED_BY_BUILDER=1
            POSTED_BY_DEALER=0
            POSTED_BY_OWNER=0
        else:
            POSTED_BY_OWNER=1
            POSTED_BY_DEALER=0
            POSTED_BY_BUILDER=0
        RERA=int(request.form["RERA"])
        BHK_NO=int(request.form["BHK_NO"])
        SQUARE_FT=float(request.form["SQUARE_FT"])
        READY_TO_MOVE =int(request.form["READY_TO_MOVE"])
        LONGTITUDE=float(request.form["LONGTITUDE"])
        LATITUDE=float(request.form["LATITUDE"])
        BHK_RK=int(request.form["BHK_RK"])
        lst=[POSTED_BY_BUILDER,POSTED_BY_DEALER,POSTED_BY_OWNER,RERA,BHK_NO,SQUARE_FT,READY_TO_MOVE,LONGTITUDE,LATITUDE,BHK_RK]
        lst_df=pd.DataFrame(np.array(lst)).transpose()
        lst_df.rename(columns={0:'POSTED_BY_Builder',1:'POSTED_BY_Dealer',2:'POSTED_BY_Owner',3:'RERA',4:'BHK_NO.',5: 'SQUARE_FT', 6:'READY_TO_MOVE', 7:'LONGITUDE',8: 'LATITUDE',9:'BHK_RK'},inplace=True)
        house_price=model.predict(lst_df)
        print(lst_df.head())
    return render_template('index.html', prediction_text=" {}".format(house_price))
    
if __name__ == "__main__":
    app.run(debug=True)