import matplotlib.pyplot as plt
from sklearn import preprocessing, svm

from prepare_data import prepare_data

def plot_ts(ts, plot_ma=True, plot_intervals=True, window=30, figsize=(15,5)):
   rolling_mean = ts.rolling(window=window).mean()
   rolling_std = ts.rolling(window=window).std()
   plt.figure(figsize=figsize)
   plt.title(ts.name)
   plt.plot(ts[window:], label='Actual values', color="black")
   if plot_ma:
      plt.plot(rolling_mean, 'g', label='MA'+str(window), color="red")
   if plot_intervals:
      lower_bound = rolling_mean - (1.96 * rolling_std)
      upper_bound = rolling_mean + (1.96 * rolling_std)
   plt.fill_between(x=ts.index, y1=lower_bound, y2=upper_bound, color='lightskyblue', alpha=0.4)
   plt.legend(loc='best')
   plt.grid(True)
   plt.show()

def one_class_SVM(data, perc=0.01, figsize=(15,5)):
    scaler = preprocessing.StandardScaler()
    ts_scaled = scaler.fit_transform(data.values.reshape(-1, 1))
    model = svm.OneClassSVM(nu=perc, kernel="rbf", gamma=0.01)
    model.fit(ts_scaled)

    df_outliers = data.to_frame(name="ts")
    df_outliers["index"] = range(len(data))
    df_outliers["outlier"] = model.predict(ts_scaled)
    df_outliers["outlier"] = df_outliers["outlier"].apply(lambda x: 1 if x==-1 else 0)

    fig, ax = plt.subplots(figsize=figsize)
    ax.set(title="Outliers detection: found " + str(sum(df_outliers["outlier"]==1)))
    ax.plot(df_outliers["index"], df_outliers["ts"], color="green")
    ax.scatter(x=df_outliers[df_outliers["outlier"]==1]["index"], y=df_outliers[df_outliers["outlier"]==1]['ts'], color='red')
    ax.grid(True)
    plt.show()

    outliers_anomaly = df_outliers[df_outliers["outlier"] == 1].T
    for index in outliers_anomaly:
        print(f'Founded anomaly date: {index}')



data = prepare_data('weatherULURU.csv')
plot_ts(data)
one_class_SVM(data)