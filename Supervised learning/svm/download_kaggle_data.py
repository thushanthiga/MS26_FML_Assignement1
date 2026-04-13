import kagglehub

# Download latest version of classical mudras
path1 = kagglehub.dataset_download("ishanishah8/indian-classical-mudras-classification")
print(f"Dataset 1 path: {path1}")

# Download latest version of kathak mudra
path2 = kagglehub.dataset_download("demon2angel/asanyukta-kathak-mudra")
print(f"Dataset 2 path: {path2}")
