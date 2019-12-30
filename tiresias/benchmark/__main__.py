from tiresias.benchmark import classification
from tiresias.benchmark import regression

df = classification.run_benchmark()
df.to_csv("classification.csv")

df = regression.run_benchmark()
df.to_csv("regression.csv")
