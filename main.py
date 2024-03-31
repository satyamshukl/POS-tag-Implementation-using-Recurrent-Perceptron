import sys
import os

def run_python_file(file_path):
    if os.path.exists(file_path) and file_path.endswith('.py'):
        try:
            os.system(f"python3 {file_path}")
        except Exception as e:
            print(f"An error occurred while running the file: {e}")
    else:
        print("Invalid file path or file is not a Python file.")

def main():
    print("Choose a Python file to run:")
    print("1. train.py")
    print("2. cross_val_train.py")
    choice = input("Enter the number corresponding to your choice: ")
    print()
    print('Starting Training...')

    if choice == "1":
        run_python_file("train.py")
    elif choice == "2":
        run_python_file("cross_val_train.py")
    else:
        print("Invalid choice. Please enter either '1' or '2'.")

if __name__ == "__main__":
    main()
