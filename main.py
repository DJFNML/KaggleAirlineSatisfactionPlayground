import kagglehub

kagglehub.login()  # Login to Kaggle
# Download latest version
path = kagglehub.competition_download('playground-series-s6e10')

print("Path to competition files:", path)