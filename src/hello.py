import pandas as pd
import random

rows = 999999
data = []

for _ in range(rows):
    temperature = random.randint(15, 40)
    humidity = random.randint(20, 100)
    pressure = random.randint(980, 1030)
    windspeed = random.randint(0, 25)
    cloudcover = random.randint(0, 100)

    rain = 1 if (humidity > 75 and cloudcover > 60 and pressure < 1010) else 0

    data.append([
        temperature,
        humidity,
        pressure,
        windspeed,
        cloudcover,
        rain
    ])

df = pd.DataFrame(data, columns=[
    "Temperature",
    "Humidity",
    "Pressure",
    "WindSpeed",
    "CloudCover",
    "RainTomorrow"
])

df.to_csv("rainfall.csv", index=False)

print("rainfall.csv generated with 1000 rows")
