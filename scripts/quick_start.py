#!/usr/bin/env python3
"""
Quick start script for Autonomous Driving System
Provides easy demo and testing functionality
"""
import os
import sys
import subprocess
import argparse
import webbrowser

# Add src to path
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'src'))

def print_banner():
    """Print welcome banner"""
    banner = """
    ╔══════════════════════════════════════════════════════╗
    ║            🚗 Autonomous Driving System              ║
    ║                                                      ║
    ║  Quick Start Demo - Choose your adventure!          ║
    ╚══════════════════════════════════════════════════════╝
    """
    print(banner)

def check_dependencies():
    """Check if required dependencies are installed"""
    required_packages = ['tensorflow', 'opencv-python', 'numpy', 'flask']
    missing = []
    
    for package in required_packages:
        try:
            __import__(package.replace('-', '_'))
        except ImportError:
            missing.append(package)
    
    if missing:
        print(f"❌ Missing required packages: {', '.join(missing)}")
        print("   Run: pip install -r requirements.txt")
        return False
    
    print("✅ All required packages are installed!")
    return True

def demo_webcam_lane_detection():
    """Demo webcam lane detection"""
    print("\n🎥 Starting webcam lane detection demo...")
    print("   Press 'q' to quit")
    
    cmd = [
        sys.executable, "drive_realtime.py",
        "--approach", "lane",
        "--interface", "webcam",
        "--duration", "30"
    ]
    
    subprocess.run(cmd)

def demo_cnn_prediction():
    """Demo CNN model prediction on sample images"""
    print("\n🧠 Testing CNN model on sample images...")
    
    # Check if model exists
    model_path = os.path.join("models", "cnn_model.h5")
    if not os.path.exists(model_path):
        print(f"❌ CNN model not found at {model_path}")
        print("   You need to train a model first using: python train_cnn.py")
        return
    
    cmd = [
        sys.executable, "test_model.py",
        "--cnn-model", model_path,
        "--webcam",
        "--duration", "30"
    ]
    
    subprocess.run(cmd)

def start_simulator_mode():
    """Start simulator interface"""
    print("\n🎮 Starting simulator interface...")
    print("   Make sure your simulator is running first!")
    print("   Default connection: localhost:4567")
    
    approach = input("   Choose approach (cnn/lane/hybrid) [cnn]: ").strip() or "cnn"
    
    cmd = [
        sys.executable, "drive_realtime.py",
        "--approach", approach,
        "--interface", "simulator"
    ]
    
    if approach == "cnn":
        model_path = os.path.join("models", "cnn_model.h5") 
        if os.path.exists(model_path):
            cmd.extend(["--model", model_path])
        else:
            print(f"⚠️  Warning: CNN model not found at {model_path}")
    
    subprocess.run(cmd)

def train_new_model():
    """Train a new CNN model"""
    print("\n🏋️  Training new CNN model...")
    
    # Check if training data exists
    data_dir = "data"
    if not os.path.exists(os.path.join(data_dir, "driving_log.csv")):
        print(f"❌ Training data not found in {data_dir}/")
        print("   Please ensure you have:")
        print("   - data/driving_log.csv")
        print("   - data/IMG/ directory with images")
        return
    
    epochs = input("   Number of epochs [10]: ").strip() or "10"
    
    cmd = [
        sys.executable, "train_cnn.py",
        "--epochs", epochs,
        "--data-dir", data_dir
    ]
    
    subprocess.run(cmd)

def show_performance_test():
    """Show performance testing options"""
    print("\n📊 Performance Testing Options:")
    print("   1. Test CNN model accuracy")
    print("   2. Real-time FPS test")
    print("   3. Compare CNN vs Lane Detection")
    
    choice = input("   Choose test (1-3): ").strip()
    
    if choice == "1":
        model_path = input("   CNN model path [models/cnn_model.h5]: ").strip() or "models/cnn_model.h5"
        test_data = input("   Test data CSV [data/driving_log.csv]: ").strip() or "data/driving_log.csv"
        
        cmd = [
            sys.executable, "test_model.py",
            "--cnn-model", model_path,
            "--test-data", test_data
        ]
        subprocess.run(cmd)
        
    elif choice == "2":
        print("   Testing real-time performance with webcam...")
        cmd = [
            sys.executable, "test_model.py",
            "--webcam",
            "--duration", "30"
        ]
        subprocess.run(cmd)
        
    elif choice == "3":
        model_path = input("   CNN model path [models/cnn_model.h5]: ").strip() or "models/cnn_model.h5"
        test_images = input("   Test images directory [data/IMG]: ").strip() or "data/IMG"
        
        cmd = [
            sys.executable, "test_model.py",
            "--cnn-model", model_path,
            "--test-images", test_images,
            "--compare"
        ]
        subprocess.run(cmd)

def open_documentation():
    """Open documentation in browser"""
    readme_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "README.md")
    if os.path.exists(readme_path):
        webbrowser.open(f"file://{os.path.abspath(readme_path)}")
    else:
        print("📖 README.md not found")

def main():
    parser = argparse.ArgumentParser(description='Quick start for Autonomous Driving System')
    parser.add_argument('--check-only', action='store_true', help='Only check dependencies')
    args = parser.parse_args()
    
    print_banner()
    
    # Check dependencies
    if not check_dependencies():
        return
    
    if args.check_only:
        return
    
    while True:
        print("\n🚀 Quick Start Options:")
        print("   1. 🎥 Demo webcam lane detection")
        print("   2. 🧠 Test CNN model predictions")
        print("   3. 🎮 Start simulator mode")
        print("   4. 🏋️  Train new CNN model")
        print("   5. 📊 Performance testing")
        print("   6. 📖 Open documentation")
        print("   0. 🚪 Exit")
        
        choice = input("\n   Choose option (0-6): ").strip()
        
        if choice == "0":
            print("👋 Goodbye!")
            break
        elif choice == "1":
            demo_webcam_lane_detection()
        elif choice == "2":
            demo_cnn_prediction()
        elif choice == "3":
            start_simulator_mode()
        elif choice == "4":
            train_new_model()
        elif choice == "5":
            show_performance_test()
        elif choice == "6":
            open_documentation()
        else:
            print("❌ Invalid choice. Please try again.")
        
        input("\n   Press Enter to continue...")

if __name__ == "__main__":
    main() 