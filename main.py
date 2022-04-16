# %%
import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as matplotlib
import statsmodels.formula.api as smf
import pmdarima as pm

# %%
from pathlib import Path

# %%
plt.style.use("seaborn-darkgrid")
matplotlib.rcParams["axes.labelsize"] = 14
matplotlib.rcParams["xtick.labelsize"] = 12
matplotlib.rcParams["ytick.labelsize"] = 12
matplotlib.rcParams["text.color"] = "k"
matplotlib.rcParams["figure.dpi"] = 200

# %%
directory = "plots"
Path(directory).mkdir(parents=True, exist_ok=True)

# %% Ex 1
sns.get_dataset_names()
diamonds = sns.load_dataset(
    "diamonds"
)  # total per billions miles, next four columns are %
print(diamonds)

# %%
diamonds.columns

# %%
diamonds["volume"] = diamonds["x"] * diamonds["y"] * diamonds['z']
# %%
print(diamonds)
# %%
diamonds.price.describe()
# %%
plt.hist(
    x=[diamonds.price],
    bins=20,
    label=["price"],
    color=["darkred"],
    rwidth=1,
)
plt.legend()
plt.savefig(directory + "/hist_price.png")

# %%
def LeastSquares(xs, ys):
    mean_x = np.mean(xs)
    var_x = np.var(xs)
    mean_y = np.mean(ys)
    cov = np.dot(xs - mean_x, ys - mean_y) / len(xs)
    slope = cov / var_x
    inter = mean_y - slope * mean_x
    return inter, slope


# %%
inter, slope = LeastSquares(diamonds.carat, diamonds.price)
print(f"Carat: {inter=}; {slope=}")
diamonds["fit_carat"] = inter + slope * diamonds["carat"]

# %%
plt.scatter(x="carat", y="price", data=diamonds, color="blue", label="carat", alpha=0.5)
plt.scatter(x="carat", y="fit_carat", data=diamonds, color="cyan", marker="s", alpha=0.5)
plt.xlabel("carat")
plt.ylabel("price")
plt.legend()
plt.savefig(f"{directory}/scatter.png")
# %% Ex 2

planets = sns.load_dataset("planets")
# %%
planets["twentyfirst_century"] = (planets.year > 2000) * 1
# %%
model = smf.logit(
    "twentyfirst_century ~ orbital_period + mass + distance", data=planets
)
results = model.fit()
results.summary()
# %%
new_planet = pd.DataFrame(
    data={"orbital_period": [100], "mass": [1], "distance": [100]}
)
# %%
y = results.predict(new_planet)
# %%
print(
    f"the chances of discovering such a planet in the 21st c. (vs in the 20th) are {y[0]}"
)

# %% Ex 3
air = pd.read_csv(".lesson/assets/AirQualityUCI.csv", sep=";")
air["Date"] = pd.to_datetime(air["Date"])
air = air.sort_values("Date").query('Date > "2004-04-01"').query('Date < "2004-12-31"')
smooth_air = air.copy()
smooth_air["NOx(GT)"] = air["NOx(GT)"].ewm(span=30).mean()
# %%
plt.plot(air["Date"], air["NOx(GT)"], label="nox", color="gray")
plt.plot(smooth_air["Date"], smooth_air["NOx(GT)"], label="nox ewma", color="darkred")
plt.legend()
plt.savefig("plots/ewma.png")
# %%
pd.plotting.autocorrelation_plot(smooth_air["NOx(GT)"])
# %%
with pm.StepwiseContext(max_dur=15):
    model = pm.auto_arima(smooth_air["NOx(GT)"], stepwise=True, error_action="ignore")
results = model.fit(smooth_air["NOx(GT)"])
print(results.summary())
# %%
