import pandas as pd
import os

train_path = "pp5i_train.gr.csv"
test_path = "pp5i_test.gr.csv"
class_path = "pp5i_train_class.txt"

print(f"Current directory: {os.getcwd()}")
print(f"Files in directory: {os.listdir('.')}")

try:
    print("Loading train...")
    train_df = pd.read_csv(train_path)
    print(f"Train shape: {train_df.shape}")

    print("Loading test...")
    test_df = pd.read_csv(test_path)
    print(f"Test shape: {test_df.shape}")

    print("Loading class...")
    class_df = pd.read_csv(class_path)
    print(f"Class shape: {class_df.shape}")

    print("Done!")
except Exception as e:
    import traceback
    traceback.print_exc()
