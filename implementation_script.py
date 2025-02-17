import joblib
import pandas as pd
import matplotlib.pyplot as plt
from neuralprophet import NeuralProphet
import datetime

def get_forecast_duration():
    number_of_days = 0
    options = {
        "1": 1, "2": 3, "3": 7, "4": 30, "5": 90,
        "6": 180, "7": 270, "8": 365
    }
    print("Select forecast duration:")
    print("1: One day\n2: Three days\n3: One week\n4: One month\n5: Three months\n6: Six months\n7: Nine months\n8: One year\n9: Custom")
    choice = int(input("Enter choice (1 to 9): "))
    if choice == 1:
        number_of_days = 1
    elif choice == 2:
        number_of_days = 3
    elif choice == 3:
        number_of_days = 7
    elif choice == 4:
        number_of_days = 30
    elif choice == 5:
        number_of_days = 90
    elif choice == 6:
        number_of_days = 180
    elif choice == 7:
        number_of_days = 270
    elif choice == 8:
        number_of_days = 365
    elif choice == 9:
        number_of_days = int(input("Enter the forecast duration in days (numbers only): "))

lagos_model = joblib.load("Lagos Flood Prediction Model.pkl")
ilorin_model = joblib.load("Ilorin Flood Prediction Model.pkl")

future_periods = get_forecast_duration()

lagos_future = lagos_model.make_future_dataframe(pd.DataFrame(columns=["ds", "y"]), periods=number_of_days)
ilorin_future = ilorin_model.make_future_dataframe(pd.DataFrame(columns=["ds", "y"]), periods=number_of_days)

lagos_pred = lagos_model.predict(lagos_future)
ilorin_pred = ilorin_model.predict(ilorin_future)

fig, ax = plt.subplots(2, 1, figsize=(12, 10))
lagos_model.plot(lagos_pred, ax=ax[0])
ax[0].set_title("Lagos Precipitation Forecast")
ax[0].set_xlabel("Date")
ax[0].set_ylabel("Precipitation (mm)")

ilorin_model.plot(ilorin_pred, ax=ax[1])
ax[1].set_title("Ilorin Precipitation Forecast")
ax[1].set_xlabel("Date")
ax[1].set_ylabel("Precipitation (mm)")

plt.tight_layout()
plt.show()

timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
lagos_pred.to_csv(f"Lagos_Precipitation_Forecast_{timestamp}.csv", index=False)
ilorin_pred.to_csv(f"Ilorin_Precipitation_Forecast_{timestamp}.csv", index=False)

print("Predictions saved successfully!")
