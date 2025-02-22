from django.shortcuts import render
from django.http import JsonResponse
from django.http import HttpResponse
# Create your views here.
from django.contrib.staticfiles.storage import staticfiles_storage
import numpy as np
import pandas as pd
import requests, json
from . import static


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier

import numpy as np
import pandas as pd
import joblib
from django.shortcuts import render
from django.http import JsonResponse
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import r2_score


def recommend_helper():

    #train_data = staticfiles_storage.path('stati')
    #train_data = pd.read_csv("{%static mysite/files/crop.csv %}")
    train_data = pd.read_csv("https://raw.githubusercontent.com/dphi-official/Datasets/master/crop_recommendation/train_set_label.csv")

    le = LabelEncoder()
    train_data.crop = le.fit_transform(train_data.crop)

    X = train_data.drop('crop', axis=1)
    y = train_data['crop']

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.3)

    model = RandomForestClassifier(n_estimators=150)

    model.fit(X_train, y_train)

    return model, le


def recommend(N, P, K, T, H, ph, R):
    A = [N, P, K, T, H, ph, R]

    model, le = recommend_helper()

    S = np.array(A)
    X = S.reshape(1, -1)

    pred = model.predict(X)

    crop_pred = le.inverse_transform(pred)

    return crop_pred[0]

def getCityInfo(city_name):
    api_key = "15e46bb2ab66ccd2c49c545973237381"
    base_url = "http://api.openweathermap.org/data/2.5/weather?"
    #city_name = input("Enter city name : ")
    complete_url = base_url + "appid=" + api_key + "&q=" + city_name
    response = requests.get(complete_url)
    x = response.json()
    print(x)
    if x["cod"] != "404":
        y = x["main"]
        current_temperature = y["temp"]-273.15
        current_pressure = y["pressure"]
        current_humidiy = y["humidity"]
        z = x["weather"]
        weather_description = z[0]["description"]
        #print(current_temperature,current_pressure,current_humidiy,weather_description)
        return current_temperature,current_humidiy,current_temperature+100





def index(request):
    return render(request,'mysite/index.html')


def about(request):
    return render(request,'mysite/about.html')

def prediction(request):
    return render(request,'mysite/predict.html')

def yprediction(request):
    return render(request,'mysite/ypredict.html')

def frecommend(request):
    return render(request,'mysite/frecommend.html')

def crecommend(request):
    return render(request,'mysite/crecommend.html')


def schemes(request):
    return render(request,'mysite/schemes.html')


def livefeedpage(request):
    return render(request,'mysite/404.html')
def community(request):
    return render(request,'mysite/404.html')

def contact(request):
    return render(request,'mysite/contact.html')

def index1(request):
    if request.method == 'POST':
        try:
            dataa = json.loads(request.POST['content'] )
            print(dataa)
            t1,t2,t3=getCityInfo(dataa[0])

            crop1=recommend(dataa[2],dataa[3],dataa[4], t1, t2, dataa[1], t3)
            print(crop1)
        except:
            crop1="Invalid Input"
            print("Invalid Input")
        return JsonResponse({'message': 'success', 'username': "username", 'content': crop1})
    


def yield_train_model():

    df = pd.read_csv("https://raw.githubusercontent.com/Sanjana9211/Datasets/refs/heads/main/crop_production_karnataka.csv")

    
    # Drop the Crop_Year column
    df = df.drop(['Crop_Year'], axis=1)
    
    # Separate the features and target variable
    X = df.drop(['Production'], axis=1)
    y = df['Production']
    
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Categorical columns for one-hot encoding
    categorical_cols = ['State_Name', 'District_Name', 'Season', 'Crop']
    
    # One-hot encode the categorical columns
    ohe = OneHotEncoder(handle_unknown='ignore')
    ohe.fit(X_train[categorical_cols])
    
    # Convert categorical columns to one-hot encoding
    X_train_categorical = ohe.transform(X_train[categorical_cols])
    X_test_categorical = ohe.transform(X_test[categorical_cols])
    
    # Combine the one-hot encoded categorical columns and numerical columns
    X_train_final = np.hstack((X_train_categorical.toarray(), X_train.drop(categorical_cols, axis=1)))
    X_test_final = np.hstack((X_test_categorical.toarray(), X_test.drop(categorical_cols, axis=1)))
    
    # Train the model
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train_final, y_train)
    
    return model,ohe

    
def ypredict(state,district,season,crop,area):
    
    model,ohe=yield_train_model()
    
    
    # Prepare input for prediction
    user_input = pd.DataFrame({
        'State_Name': [state],
        'District_Name': [district],
        'Season': [season],
        'Crop': [crop],
        'Area': [area]
    })
    
    # Categorical columns for encoding
    categorical_cols = ['State_Name', 'District_Name', 'Season', 'Crop']
    
    # One-hot encode the categorical columns
    user_input_categorical = ohe.transform(user_input[categorical_cols])
    
    # Combine the one-hot encoded categorical columns and numerical columns
    user_input_final = np.hstack((user_input_categorical.toarray(), user_input[['Area']].values))
    
    # Make the prediction
    prediction = model.predict(user_input_final)
    
    return prediction[0]


def yieldcalc(request):
    if request.method == 'POST':
        try:
            dataa = json.loads(request.POST['content'] )
            print(dataa)

            crop1=ypredict(dataa[0],dataa[1],dataa[2], dataa[3],dataa[4])
            print(crop1)
        except Exception as e:
            crop1 = "Invalid Input"
            print("Error:", str(e))  
        return JsonResponse({'message': 'success', 'username': "username", 'content': crop1})


def latestnews(request):
    #url = ('https://newsapi.org/v2/top-headlines?'
    #       'sources=bbc-news&'
    #       'apiKey=cb2dbc632d8d4eefb8cbf1e87abb2a78')
    url = ('https://newsapi.org/v2/top-headlines?'
           'country=in&'
           'apiKey=cb2dbc632d8d4eefb8cbf1e87abb2a78')
    response = requests.get(url)
    l=response.json()['articles']
    #pprint(response.json()['articles'][0]['urlToImage'])
    desc = []
    news = []
    img = []

    for i in range(len(l)):
        f = l[i]
        news.append(f['title'])
        desc.append(f['description'])
        img.append(f['urlToImage'])
    mylist = zip(news, desc, img)

    return render(request, 'mysite/news.html', context={"mylist": mylist})



def fert_model():
    from sklearn.tree import DecisionTreeClassifier

    global dtc, le_soil, le_crop
    
    # Loading the dataset
    try:
        data = pd.read_csv("https://raw.githubusercontent.com/Sanjana9211/Datasets/refs/heads/main/fertilizer_recommendation.csv")
        
        # Label encoding for categorical features
        le_soil = LabelEncoder()
        data['Soil Type'] = le_soil.fit_transform(data['Soil Type'])
        le_crop = LabelEncoder()
        data['Crop Type'] = le_crop.fit_transform(data['Crop Type'])

        soil_types = le_soil.classes_  # Get original soil type names
        crop_types = le_crop.classes_ 
        
        # Splitting the data into input (X) and output (y) variables
        X = data.iloc[:, :8]
        y = data.iloc[:, -1]

        # Splitting the dataset into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

        # Training the Decision Tree Classifier model
        dtc = DecisionTreeClassifier(random_state=0)
        dtc.fit(X_train, y_train)

    except Exception as e:
        return JsonResponse({'error': str(e)})



def fertrec(request):
        if request.method == "POST":
            try:
                # Get user inputs from request
                fert_model()
                dataa = json.loads(request.POST['content'] )
                print(dataa)
                nitrogen = dataa[0]
                phosphorus = dataa[1]
                potassium = dataa[2]
                temperature = dataa[3]
                humidity = dataa[4]
                soil_moisture = dataa[5]
                soil_type = dataa[6]
                crop_type = dataa[7]
                # Encode categorical values
                soil_encoded = le_soil.transform([soil_type])[0]
                crop_encoded = le_crop.transform([crop_type])[0]

                # Prepare input for model prediction
                user_input = np.array([[temperature, humidity, soil_moisture, soil_encoded, crop_encoded, nitrogen, potassium, phosphorus]])

                # Predict the fertilizer recommendation
                prediction = dtc.predict(user_input)[0]

                # Return JSON response with prediction
                return JsonResponse({'message': 'success', 'username': "username", 'content': prediction})

            except Exception as e:
                return JsonResponse({'error': str(e)})

        return JsonResponse({'error': 'Invalid request'})
    


def crop_model():
    import numpy as np
    import pandas as pd
    from django.shortcuts import render
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score

    # Load dataset
    dataset = pd.read_csv("https://raw.githubusercontent.com/Sanjana9211/Datasets/refs/heads/main/Crop_recommendation.csv")

    # Split features and labels
    X = dataset.iloc[:, :-1].values
    y = dataset.iloc[:, -1].values

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

    # Train Random Forest model
    classifier = RandomForestClassifier(n_estimators=10, criterion="entropy", random_state=0)
    classifier.fit(X_train, y_train)
    return classifier

    
def croprec(request):
    if request.method == "POST":
        try:
            # Retrieve user inputs from POST request
            classifier=crop_model()
            dataa = json.loads(request.POST['content'] )
            print(dataa)
            nitrogen = dataa[0]
            phosphorus = dataa[1]
            potassium = dataa[2]
            temperature = dataa[3]
            humidity = dataa[4]
            ph=dataa[5]
            rainfall=dataa[6]

            # Convert inputs to NumPy array
            user_input = np.array([[nitrogen, phosphorus, potassium, temperature, humidity, ph, rainfall]])

            # Make prediction
            prediction = classifier.predict(user_input)[0]

            return JsonResponse({'message': 'success', 'username': "username", 'content': prediction})

        except Exception as e:
            return JsonResponse({'error': str(e)})

    return JsonResponse({'error': 'Invalid request'})
    