# %%
import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as matplotlib
import statsmodels.formula.api as smf

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

mpg = sns.load_dataset("mpg")
# %%
mpg["usa"] = (mpg.origin == "usa") * 1
# %%
model = smf.logit(
    "usa ~ weight + cylinders", data=mpg
)
results = model.fit()
results.summary()
# %%
new_car = pd.DataFrame(
    data={"weight": [3000], "cylinders": [4]}
)
# %%
y = results.predict(new_car)
# %%
print(
    f"the chances of this car being made in the US (vs other parts of the world) are {y[0]}"
)

# %% Ex 3
temp = pd.read_csv(".lesson/assets/daily-min-temperatures.csv", sep=",")
temp["date"] = pd.to_datetime(temp["date"])
temp = temp.sort_values("date").query('date > "1982-01-01"').query('date < "1989-12-31"')
smooth_temp = temp.copy()
smooth_temp["temp"] = temp["temp"].ewm(span=30).mean()
# %%
plt.plot(temp["date"], temp["temp"], label="temp", color="gray")
plt.plot(smooth_temp["date"], smooth_temp["temp"], label="temp ewma", color="darkred")
plt.legend()
plt.savefig("plots/ewma.png")
# %%
mooth_temp = temp.copy()
smooth_temp["temp"] = temp["temp"].ewm(span=30).mean()
# %%
plt.plot(temp["date"], temp["temp"], label="temp", color="gray")
plt.plot(smooth_temp["date"], smooth_temp["temp"], label="temp rolling", color="darkred")
plt.legend()
plt.savefig("plots/rolling.png")
# %%
