# Microsoft Virtual Hackathon - Grab Estimated Time Arrival

## Clustering
Weighted KMeans of density (total unique trip per lat, lng) and speed (median speed per lat lng)

## Meta origin-destination
With a limited records -- 10,000 trips, we upsample by extracting every 60-seconds window as meta origin-destination pair. However, we get a heavily skewed result to the lower ends due to imbalance of data points, we place a filter where duration >= 600 seconds and distance >= 1km.

## Features
- Day (day_of_week and its sin, cos equivalent)
- Hour (hour_of_day and its sin, cos equivalent)
- Origin
  - distance from speed/density cluster center
  - bearing from speed/density cluster center
- Destination
  - distance from speed/density cluster center
  - bearing from speed/density cluster center
- Origin-Destination-Distance
- Origin-Destination-Bearing
- Origin-Destination speed cluster centers distance
- Origin-Destination density cluster centers distance
- Squared distance features
- Feature-crossing and binning of day-and-hour
- Feature-crossing and binning of origin-density-cluster and destination-density-cluster
- Feature-crossing and binning of origin-speed-cluster and destination-speed-cluster

## Models
- AutoML - StackEnsemble (Experiment RMSE ~ 252)
- Fastai - DNN (RMSE ~ 260)

We decided to move forward with Fastai DNN for our obsession with deep learning and we believe that it generalizes better based on the result distribution.

## Endpoint
To try it out:
```
curl --location --request POST 'vanagrab1.southeastasia.azurecontainer.io/grab/eta/dnn' \
--header 'Content-Type: application/json' \
--data-raw '{
    "latitude_origin": 1.340839,
    "longitude_origin": 103.848598,
    "timestamp": 1554850471,
    "day_of_week": 2,
    "hour_of_day": 22,
    "latitude_destination": 1.3448060250658491,
    "longitude_destination": 103.9831013487792
}'
```
You should receive JSON result:
```
{"eta": 893}
```

In the event if our trial credits are exhausted or expired, kindly find the source code and models in app folder.

Update - Please try replacing the above endpoint to https://vanagrab.jjkoh.com/grab/eta/dnn

## Disclaimer
This repository is not meant for repeated from end to end. The repository is extremely unorganised as most works are iterated ad hoc and spontaneously. On top of that, a majority part of the code is run on GCP. However, we try to use open-source tools as much as possible to make this solution cloud-neutral.
