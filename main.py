import os
import torch

def check_environment():
    print("✅ Python is working")
    print(f"📁 Current working directory: {os.getcwd()}")
    print(f"📦 Torch version: {torch.__version__}")
    print(f"🧠 CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"🖥️ CUDA device: {torch.cuda.get_device_name(0)}")

if __name__ == "__main__":
    check_environment()