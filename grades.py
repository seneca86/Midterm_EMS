# %%
import pandas as pd
import plotly.express as px

# %%
df = pd.read_csv('grades.csv')

fig = px.histogram(df, x='grade', nbins=20)
fig.write_image('histogram.png')

# %%
mu = df.grade.mean()
sigma = df.grade.std()
# %%
mu_up = 70
sigma_up = 16
df_up = (
    df
    .assign(t = lambda x: (x.grade - mu) / sigma)
    .assign(grade_up = lambda x: x.t * sigma_up + mu_up)
    .assign(delta = lambda x: (x.grade_up - x.grade).round())
    .assign(grade_up = lambda x: (x.grade_up).round())
    .assign(t = lambda x: x.t.round())
)

# %%
fig = px.histogram(df_up, x='grade_up', nbins=20)
fig.write_image('histogram_up.png')
df_up.to_csv('grades_up.csv', index=False)

# %%
