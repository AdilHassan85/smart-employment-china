import subprocess
import sys

steps = [
    ("Data cleaning", "src/data/clean_data.py"),
    ("Feature engineering", "src/features/build_features.py"),
    ("Model training", "src/models/train.py"),
]

for name, script in steps:
    print(f"\n{'='*50}\n{name}\n{'='*50}")
    result = subprocess.run([sys.executable, script])
    if result.returncode != 0:
        print(f"\n{name} FAIL ho gaya. Upar error dekho.")
        sys.exit(1)

print("\nPoori pipeline successfully chal gayi. Models 'models/' folder mein ready hain.")