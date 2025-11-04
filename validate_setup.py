"""
Quick validation script to check project setup.
"""

import os
from pathlib import Path

print("🔍 Validating Project Setup...\n")

# Check required files
required_files = [
    "app.py",
    "requirements.txt",
    "Dockerfile",
    "start.sh",
    "README.md"
]

print("📁 Checking required files:")
all_present = True
for file in required_files:
    exists = Path(file).exists()
    status = "✅" if exists else "❌"
    print(f"  {status} {file}")
    if not exists:
        all_present = False

# Check data directory
data_dir = Path("data")
print(f"\n📂 Data directory: {'✅ exists' if data_dir.exists() else '❌ missing'}")
if data_dir.exists():
    csv_files = list(data_dir.glob("*.csv"))
    print(f"  CSV files: {len(csv_files)}")
    if csv_files:
        print(f"  Sample: {csv_files[0].name}")

# Check Python syntax
print("\n🐍 Checking Python syntax:")
try:
    with open("app.py", "r") as f:
        code = f.read()
    compile(code, "app.py", "exec")
    print("  ✅ app.py syntax is valid")
except SyntaxError as e:
    print(f"  ❌ Syntax error: {e}")
    all_present = False

# Check requirements.txt
print("\n📦 Checking requirements.txt:")
if Path("requirements.txt").exists():
    with open("requirements.txt", "r") as f:
        reqs = f.readlines()
    print(f"  ✅ {len([r for r in reqs if r.strip() and not r.startswith('#')])} dependencies listed")

# Summary
print("\n" + "="*50)
if all_present:
    print("✅ Project setup looks good!")
    print("\n🚀 To run the application:")
    print("   1. Install dependencies: pip install -r requirements.txt")
    print("   2. Generate sample data: python generate_sample_data.py")
    print("   3. Run app: python app.py")
    print("\n🐳 For Docker deployment:")
    print("   1. Build: docker build -t churn-causal-analysis .")
    print("   2. Run: docker run -p 8000:8000 -p 8501:8501 churn-causal-analysis")
else:
    print("❌ Some required files are missing")
print("="*50)

