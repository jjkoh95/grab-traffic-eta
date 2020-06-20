from flask import Flask, request, Response, jsonify
import os
import pickle
from sklearn.cluster import KMeans
from math import radians, cos, sin, asin, sqrt, pi
from geographiclib.geodesic import Geodesic
import datetime
from fastai.tabular import *
import requests
import json

app = Flask(__name__)
app.model = None
app.speed_kmeans = None
app.density_kmeans = None


def load_kmeans():
    with open('./traffic-cluster-density-20-kmeans.pkl', 'rb') as pkl:
        app.density_kmeans = pickle.load(pkl)
    with open('./traffic-cluster-speed-20-kmeans.pkl', 'rb') as pkl:
        app.speed_kmeans = pickle.load(pkl)

def haversine(lat1, lon1, lat2, lon2):
    """
    Calculate the great circle distance between two points
    on the earth (specified in decimal degrees)
    """
    # convert decimal degrees to radians
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])

    # haversine formula
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a))
    r = 6371  # Radius of earth in kilometers. Use 3956 for miles
    return c * r

def get_bearing(lat1, long1, lat2, long2):
    brng = Geodesic.WGS84.Inverse(lat1, long1, lat2, long2)['azi1']
    return brng

def rename_body(body):
    return {"origin_lat": body["latitude_origin"],
            "origin_lng": body["longitude_origin"],
            "dest_lat": body["latitude_destination"],
            "dest_lng": body["longitude_destination"],
            "timestamp": body["timestamp"],
            "hour_of_day": body["hour_of_day"],
            "day_of_week": body["day_of_week"],
            }

def preproc(row, c):
    """Expect {origin_lat, origin_lng, origin_timestamp, dest_lat, dest_lng}"""
    row['origin_day'] = int(row['day_of_week'])
    row['origin_hour'] = int(row['hour_of_day'])
    row['origin_day_sin'] = round(sin(row['origin_day']*(pi/7)), 4)
    row['origin_day_cos'] = round(cos(row['origin_day']*(pi/7)), 4)
    row['origin_hour_sin'] = round(sin(row['origin_hour']*(pi/24)), 4)
    row['origin_hour_cos'] = round(cos(row['origin_hour']*(pi/24)), 4)

    row['origin_density_cluster'] = int(c['density'].predict([[row['origin_lat'], row['origin_lng']]])[0])
    row['origin_speed_cluster'] = int(c['speed'].predict([[row['origin_lat'], row['origin_lng']]])[0])
    origin_density_center = c['density'].cluster_centers_[int(row['origin_density_cluster'])]
    origin_speed_center = c['speed'].cluster_centers_[int(row['origin_speed_cluster'])]

    row['origin_distance_from_density_center'] = haversine(row['origin_lat'], row['origin_lng'], origin_density_center[0], origin_density_center[1])
    row['origin_bearing_from_density_center'] = get_bearing(row['origin_lat'], row['origin_lng'], origin_density_center[0], origin_density_center[1])
    row['origin_distance_from_speed_center'] = haversine(row['origin_lat'], row['origin_lng'], origin_speed_center[0], origin_speed_center[1])
    row['origin_bearing_from_speed_center'] = get_bearing(row['origin_lat'], row['origin_lng'], origin_speed_center[0], origin_speed_center[1])

    # dest data stuffs
    row['dest_density_cluster'] = int(c['density'].predict([[row['dest_lat'], row['dest_lng']]])[0])
    row['dest_speed_cluster'] = int(c['speed'].predict([[row['dest_lat'], row['dest_lng']]])[0])
    dest_density_center = c['density'].cluster_centers_[int(row['dest_density_cluster'])]
    dest_speed_center = c['speed'].cluster_centers_[int(row['dest_speed_cluster'])]

    row['dest_distance_from_density_center'] = haversine(row['dest_lat'], row['dest_lng'], dest_density_center[0], dest_density_center[1])
    row['dest_bearing_from_density_center'] = get_bearing(row['dest_lat'], row['dest_lng'], dest_density_center[0], dest_density_center[1])
    row['dest_distance_from_speed_center'] = haversine(row['dest_lat'], row['dest_lng'], dest_speed_center[0], dest_speed_center[1])
    row['dest_bearing_from_speed_center'] = get_bearing(row['dest_lat'], row['dest_lng'], dest_speed_center[0], dest_speed_center[1])

    # origin-destination stuffs
    row['distance_origin_dest'] = haversine(row['origin_lat'], row['origin_lng'], row['dest_lat'], row['dest_lng'])
    row['bearing_origin_dest'] = get_bearing(row['origin_lat'], row['origin_lng'], row['dest_lat'], row['dest_lng'])

    row['distance_origin_dest_density_cluster'] = haversine(origin_density_center[0], origin_density_center[1], dest_density_center[0], dest_density_center[1])
    row['distance_origin_dest_speed_cluster'] = haversine(origin_speed_center[0], origin_speed_center[1], dest_speed_center[0], dest_speed_center[1])

    row['origin_distance_from_density_center_squared'] = round(row['origin_distance_from_density_center'] ** 2, 4)
    row['origin_distance_from_speed_center_squared'] = round(row['origin_distance_from_speed_center'] ** 2, 4)
    row['dest_distance_from_density_center_squared'] = round(row['dest_distance_from_density_center'] ** 2, 4)
    row['dest_distance_from_speed_center_squared'] = round(row['dest_distance_from_speed_center'] ** 2, 4)
    row['distance_origin_dest_squared'] = round(row['distance_origin_dest'] ** 2, 4)
    row['distance_origin_dest_density_cluster_squared'] = round(row['distance_origin_dest_density_cluster'] ** 2, 4)
    row['distance_origin_dest_speed_cluster_squared'] = round(row['distance_origin_dest_speed_cluster'] ** 2, 4)

    row['day_hour_crossing'] = int(row['origin_day']*24 + row['origin_hour'])
    row['origin_dest_density_cluster_crossing'] = int(row['origin_density_cluster']*20 + row['dest_density_cluster'])
    row['origin_dest_speed_cluster_crossing'] = int(row['origin_speed_cluster']*20 + row['dest_speed_cluster'])

    return row

def load_model():
    app.model = load_learner('./', 'model.pkl')

@app.route('/')
def index():
    return "Hello world =D"

@app.route('/grab/eta/dnn', methods=['POST'])
def dnn():
    if not app.speed_kmeans or app.density_kmeans:
        load_kmeans()
    if not app.model:
        load_model()
    body = rename_body(request.get_json())
    preprocessed_data = preproc(body, { 'density': app.density_kmeans, 'speed': app.speed_kmeans })
    res = app.model.predict(preprocessed_data)
    return jsonify({"eta": int(res[1].cpu().numpy()[0])})

@app.route('/grab/eta/stackensemble', methods=['POST'])
def stack_ensemble():
    if not app.speed_kmeans or app.density_kmeans:
        load_kmeans()
    if not app.model:
        load_model()
    body = rename_body(request.get_json())
    preprocessed_data = preproc(body, { 'density': app.density_kmeans, 'speed': app.speed_kmeans })
    url = 'http://962e4fc9-85e8-4151-8436-892581ce696e.eastus2.azurecontainer.io/score'
    preprocessed_data.pop('origin_lat', None)
    preprocessed_data.pop('origin_lng', None)
    preprocessed_data.pop('origin_timestamp', None)
    preprocessed_data.pop('timestamp', None)
    preprocessed_data.pop('day_of_week', None)
    preprocessed_data.pop('hour_of_day', None)
    preprocessed_data.pop('dest_lat', None)
    preprocessed_data.pop('dest_lng', None)
    res = requests.post(url, json={'data': [preprocessed_data]})
    parsed_res = json.loads(res.json())['result'][0] # Azure AutoML returns raw string
    return jsonify({'eta': int(parsed_res)})

@app.route('/grab/eta/preprocessing', methods=['POST'])
def preprocessing():
    if not app.speed_kmeans or app.density_kmeans:
        load_kmeans()
    if not app.model:
        load_model()
    body = rename_body(request.get_json())
    preprocessed_data = preproc(body, { 'density': app.density_kmeans, 'speed': app.speed_kmeans })
    return jsonify(preprocessed_data)


if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=8080)
