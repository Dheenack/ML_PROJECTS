# necessary imports
from flask import Flask, request, render_template, send_file, redirect, url_for, jsonify
import pandas as pd
import xgboost as xgb
import joblib
import os
import json
import psutil
import time
import subprocess
from datetime import datetime

# model and label encoder loading
model = joblib.load('models/xgboost_model.h5')
le = joblib.load('models/label_encoder.pkl')

#Flask app
app = Flask(__name__)

#anomaly detection function
def detect_anomalies(csv_file_path):
    # Load the dataset
    data = pd.read_csv(csv_file_path, encoding="latin-1")
    
    # Preprocess data
    data['maxUE_UL+DL'] = data['maxUE_UL+DL'].replace('#¡VALOR!', pd.NA)
    data['maxUE_UL+DL'] = pd.to_numeric(data['maxUE_UL+DL'], errors='coerce')
    
    # Encode 'CellName'
    if 'CellName' in data.columns:
        data['CellName_encoded'] = le.transform(data['CellName'])
    else:
        raise ValueError("The 'CellName' column is missing from the input data.")

    # Convert 'Time' to datetime and extract features
    if 'Time' in data.columns:
        data['Time'] = pd.to_datetime(data['Time'], format='%H:%M')
        data['Hour'] = data['Time'].dt.hour
        data['DayOfWeek'] = data['Time'].dt.dayofweek
    else:
        raise ValueError("The 'Time' column is missing from the input data.")

    # Define features
    features = ['PRBUsageUL', 'PRBUsageDL', 'meanThr_DL', 'meanThr_UL', 'maxThr_DL', 'maxThr_UL', \
                'meanUE_DL', 'meanUE_UL', 'maxUE_DL', 'maxUE_UL', 'maxUE_UL+DL', 'CellName_encoded', 'Hour', 'DayOfWeek']
    
    # Check if features are present in the dataset
    missing_features = [col for col in features if col not in data.columns]
    if missing_features:
        raise ValueError(f"Missing features in data: {missing_features}")
    
    # Extract features
    X = data[features]
    
    # Predict anomalies
    data['Anomaly'] = model.predict(X)
    data['Anomaly_Probability'] = model.predict_proba(X)[:, 1]
    
    # Filter rows with anomalies
    anomalies = data[data['Anomaly'] == 1]
    
    return anomalies, data

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        file = request.files['file']
        if file:
            file_path = os.path.join('uploads', file.filename)
            file.save(file_path)
            anomalies, data = detect_anomalies(file_path)
            
            # Save CSV without anomalies
            no_anomalies_path = 'downloads/no_anomalies.csv'
            data[data['Anomaly'] == 0].to_csv(no_anomalies_path, index=False)
            
            # Store anomalies in a CSV
            anomalies.to_csv('downloads/anomalies.csv', index=False)
            
            return redirect(url_for('result', page=1))
    
    return render_template('index.html')

@app.route('/result', methods=['GET', 'POST'])
def result():
    page = int(request.args.get('page', 1))
    per_page = 100
    min_prob = float(request.args.get('min_prob', 0))
    max_prob = float(request.args.get('max_prob', 1))  # Default to 1 if max_prob is not provided
    
    anomalies_file_path = 'downloads/anomalies.csv'
    anomalies = pd.read_csv(anomalies_file_path)
    
    # Filter anomalies based on probability range
    if min_prob < max_prob:
        anomalies = anomalies[(anomalies['Anomaly_Probability'] >= min_prob) & (anomalies['Anomaly_Probability'] < max_prob)]
    
    # Reorder columns: move 'Anomaly_Probability' to the first column and drop 'Anomaly'
    anomalies = anomalies[['Anomaly_Probability'] + [col for col in anomalies.columns if col != 'Anomaly_Probability' and col != 'Anomaly' and  col != 'DayOfWeek']]

    total_entries = len(anomalies)
    start = (page - 1) * per_page
    end = start + per_page
    
    paginated_anomalies = anomalies[start:end]
    
    return render_template('result.html', 
                          anomalies=paginated_anomalies.to_html(classes='table table-striped', index=False), 
                          count=len(paginated_anomalies),
                          total_entries=total_entries,
                          current_page=page,
                          total_pages=(total_entries + per_page - 1) // per_page,
                          min_prob=min_prob,
                          max_prob=max_prob)

@app.route('/download')
def download_file():
    return send_file('downloads/no_anomalies.csv', as_attachment=True)

# Define required features (same as your model)
FEATURES = ['PRBUsageUL', 'PRBUsageDL', 'meanThr_DL', 'meanThr_UL', 'maxThr_DL', 'maxThr_UL',
            'meanUE_DL', 'meanUE_UL', 'maxUE_DL', 'maxUE_UL', 'maxUE_UL+DL', 'CellName_encoded', 'Hour', 'DayOfWeek']

@app.route('/a', methods=['GET'])
def inx():
    return render_template('a.html')


@app.route('/predict', methods=['POST','GET'])
def predict():
    try:
        # Get data from request
        def get_network_usage():
            old_stats = psutil.net_io_counters()
            time.sleep(1)  # Wait 1 second
            new_stats = psutil.net_io_counters()
    
            upload_speed = ((new_stats.bytes_sent - old_stats.bytes_sent) * 8) / 1e6
            download_speed = ((new_stats.bytes_recv - old_stats.bytes_recv) * 8) / 1e6
            return upload_speed, download_speed
        def get_active_connections():
            return len(psutil.net_connections(kind='inet'))

        def get_wifi_ssid():
            try:
                result = subprocess.check_output(["nmcli", "-t", "-f", "SSID", "dev", "wifi"], universal_newlines=True)
                return result.splitlines()[0] if result else "Unknown"
            except:
                return "Unknown"

        def get_real_time_data():
            prb_ul, prb_dl = get_network_usage()
            meanThr_DL, meanThr_UL = prb_dl, prb_ul
            maxThr_DL, maxThr_UL = max(prb_dl, 1.2), max(prb_ul, 1.0)  # Set min values
            meanUE_DL = get_active_connections()
            meanUE_UL = meanUE_DL
            maxUE_DL = meanUE_DL + 2
            maxUE_UL = meanUE_UL + 2
            maxUE_UL_DL = maxUE_DL + maxUE_UL
            cell_name = get_wifi_ssid()
            cell_name_encoded = hash(cell_name) % 100
            now = datetime.now()

            return {
                "PRBUsageUL": prb_ul,  
                "PRBUsageDL": prb_dl,  
                "meanThr_DL": meanThr_DL,  
               "meanThr_UL": meanThr_UL,
                "maxThr_DL": maxThr_DL,
               "maxThr_UL": maxThr_UL,
                "meanUE_DL": meanUE_DL,  
                "meanUE_UL": meanUE_UL,
               "maxUE_DL": maxUE_DL,
               "maxUE_UL": maxUE_UL,
               "maxUE_UL+DL": maxUE_UL_DL,
               "CellName_encoded": cell_name_encoded,
               "Hour": now.hour,
               "DayOfWeek": now.weekday()}

# Test the function
        real_time_data = get_real_time_data()
 
        df = pd.DataFrame(real_time_data, index=[0])

        # Ensure all features  are present
        missing_features = [col for col in FEATURES if col not in df.columns]
        if missing_features:
            return jsonify({"error": f"Missing features: {missing_features}"}), 400
        
        # Predict anomaly
        anomaly_prediction = model.predict(df)[0]
        anomaly_prob = model.predict_proba(df)[:, 1][0]
        
        # Return result
        #return render_template('predict.html', anomaly=int(anomaly_prediction), probability=float(anomaly_prob))
        #return jsonify({"anomaly": int(anomaly_prediction), "probability": float(anomaly_prob)}),jsonify(real_time_data)
        # Combine results
        result = {
            "anomaly": int(anomaly_prediction),
            "probability": float(anomaly_prob),
            **real_time_data  # Merge real-time data into the result
        }
        
        return jsonify(result) 
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/recent', methods=['GET'])
def recent():
    page = int(request.args.get('page', 1))
    per_page = 100
    min_prob = float(request.args.get('min_prob', 0))
    max_prob = float(request.args.get('max_prob', 1))  # Default to 1 if max_prob is not provided
    
    anomalies_file_path = 'downloads/anomalies.csv'
    anomalies = pd.read_csv(anomalies_file_path)
    
    # Filter anomalies based on probability range
    if min_prob < max_prob:
        anomalies = anomalies[(anomalies['Anomaly_Probability'] >= min_prob) & (anomalies['Anomaly_Probability'] < max_prob)]
    
    # Reorder columns: move 'Anomaly_Probability' to the first column and drop 'Anomaly'
    anomalies = anomalies[['Anomaly_Probability'] + [col for col in anomalies.columns if col != 'Anomaly_Probability' and col != 'Anomaly' and  col != 'DayOfWeek']]

    total_entries = len(anomalies)
    start = (page - 1) * per_page
    end = start + per_page
    
    paginated_anomalies = anomalies[start:end]
    return render_template('recent.html', 
                          anomalies=paginated_anomalies.to_html(classes='table table-striped', index=False), 
                          count=len(paginated_anomalies),
                          total_entries=total_entries,
                          current_page=page,
                          total_pages=(total_entries + per_page - 1) // per_page,
                          min_prob=min_prob,
                          max_prob=max_prob)

@app.route('/manual', methods=['GET','POST'])
def manual():
    return render_template('manual.html')

@app.route('/dd',methods=['POST'])
def dd():
    if not request.is_json:
        return jsonify({"error": "Content-Type must be application/json"}), 415

    data = request.get_json()

        # Convert all values to float (or int when applicable)
    try:
        for key in data:
            data[key] = float(data[key]) if '.' in str(data[key]) else int(data[key])
    except ValueError:
        return jsonify({"error": f"Invalid data format in {key}"}), 400

    df=pd.DataFrame(data=data,columns=FEATURES,index=[0])
    # Ensure all features are present
    missing_features = [col for col in FEATURES if col not in df.columns]
    if missing_features:
        return jsonify({"error": f"Missing features: {missing_features}"}), 400
    
    df = df.astype(float)  # Ensure all values are float
        # Predict anomaly
    anomaly_prediction = model.predict(df)[0]
    pred="True" if anomaly_prediction==1 else "False"
    anomaly_prob = model.predict_proba(df)[:, 1][0]

    # Render the template and send back the full HTML response
    return render_template('manual_response.html', anomaly=pred, count=anomaly_prob)
    

if __name__ == '__main__':
    if not os.path.exists('uploads'):
        os.makedirs('uploads')
    if not os.path.exists('downloads'):
        os.makedirs('downloads')
    app.run(debug=True)