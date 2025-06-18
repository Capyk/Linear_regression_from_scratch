import pandas as pd
import matplotlib.pyplot as plt


# Load the dataset
df = pd.read_csv('train.csv')

print("First 5 records:", df.head())
df = df.dropna()
print(df.describe())

# plt.scatter(df.x, df.y)
# plt.show()

def loss_function(slope, intercept, points):
    total_error = 0
    for i in range(len(points)):
        x = points.iloc[i].x
        y = points.iloc[i].y
        total_error += (y - (slope * x + intercept)) ** 2
    return total_error / float(len(points))

def R2(slope, intercept, points):
    SSR = 0
    SST = 0

    for i in range(len(points)):
        x = points.iloc[i].x
        y = points.iloc[i].y
        SSR += (y - (slope * x + intercept)) ** 2
        SST += (y - points.mean(axis=0).y) ** 2
    return 1 - (SSR/SST)


def gradient_step(current_m, current_b, data, learning_rate):
    m_delta = 0
    b_delta = 0
    total_points = len(data)

    for index, row in data.iterrows():
        x_val = row.x
        y_val = row.y
        prediction = current_m * x_val + current_b
        error = y_val - prediction

        m_delta += (-2 / total_points) * x_val * error
        b_delta += (-2 / total_points) * error

    updated_m = current_m - learning_rate * m_delta
    updated_b = current_b - learning_rate * b_delta
    return updated_m, updated_b


slope = 0
intercept = 0
learning_rate = 0.0001
iterations = 1000

for epoch in range(iterations):
    if epoch % 50 == 0:
        print(f"Epoch: {epoch}")
    slope, intercept = gradient_step(slope, intercept, df, learning_rate)


print(slope, intercept)

plt.scatter(df.x, df.y, alpha=0.3)
plt.ylim(0, 110)
plt.xlim(0, 100)
plt.plot(list(range(0, 100)), [slope * x + intercept for x in range(0, 100)], color='red')
plt.show()

df2 = pd.read_csv("test.csv")

print(loss_function(slope, intercept, df2))
print(R2(slope, intercept, df2))

