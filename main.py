import subprocess

# Run "Hardcore" KNN
subprocess.run(["python", "knn_hardcore.py"])

# Run SKLEARN KNN
subprocess.run(["python", "knn_sklearn.py"])

print("Results written on results.txt")