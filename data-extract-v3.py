import os
import lzma
import random
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import cpu_count

def process_file(args):
    """Extract text from a .xz file and return its unique characters."""
    directory, filename, output_file = args
    file_path = os.path.join(directory, filename)

    try:
        with lzma.open(file_path, "rt", encoding="utf-8") as infile:
            text = infile.read()

        with open(output_file, "a", encoding="utf-8") as outfile:
            outfile.write(text)

        return set(text)

    except Exception as e:
        print(f"[ERROR] Failed to process {filename}: {e}")
        return set()

def xz_files_in_dir(directory):
    """List all .xz files in a directory."""
    try:
        return sorted([
            f for f in os.listdir(directory)
            if f.endswith(".xz") and os.path.isfile(os.path.join(directory, f))
        ])
    except FileNotFoundError:
        print(f"[ERROR] Directory '{directory}' not found.")
        return []

def process_files_in_parallel(files, folder_path, output_file):
    """Process files in parallel and build a vocabulary set."""
    vocab = set()
    if not files:
        print(f"[WARNING] No files to process for output: {output_file}")
        return vocab

    args = [(folder_path, filename, output_file) for filename in files]

    with ProcessPoolExecutor(max_workers=cpu_count()) as executor:
        for characters in tqdm(executor.map(process_file, args), total=len(files), desc=f"Processing {output_file}"):
            vocab.update(characters)

    return vocab

def safe_sample(files, sample_rate):
    """Safely sample a subset of files based on the sample rate."""
    if not files or sample_rate <= 0:
        return []
    return random.sample(files, max(1, int(len(files) * sample_rate)))

# ---------------- CONFIG ----------------
FOLDER_PATH = "openwebtext"
OUTPUT_TRAIN = "output_train.txt"
OUTPUT_VAL = "output_val.txt"
VOCAB_FILE = "vocab.txt"
SAMPLE_RATE = 0.01

# ----------- MAIN LOGIC ---------------
if not os.path.isdir(FOLDER_PATH):
    print(f"[ERROR] Directory '{FOLDER_PATH}' does not exist.")
    exit(1)

files = xz_files_in_dir(FOLDER_PATH)
total_files = len(files)
print(f"[INFO] Found {total_files} .xz files.")

if total_files == 0:
    print("[INFO] No files to process. Exiting.")
    exit(1)

# Train/Val split
split_index = int(total_files * 0.9)
files_train = files[:split_index]
files_val = files[split_index:]
print(f"[INFO] Train files: {len(files_train)}, Val files: {len(files_val)}")

# Sampling
files_train_sampled = safe_sample(files_train, SAMPLE_RATE)
files_val_sampled = safe_sample(files_val, SAMPLE_RATE)
print(f"[INFO] Sampled Train files: {len(files_train_sampled)}, Sampled Val files: {len(files_val_sampled)}")

# Clear previous outputs
for path in [OUTPUT_TRAIN, OUTPUT_VAL]:
    with open(path, 'w', encoding='utf-8'):
        pass

# Process and collect vocab
vocab_train = process_files_in_parallel(files_train_sampled, FOLDER_PATH, OUTPUT_TRAIN)
vocab_val = process_files_in_parallel(files_val_sampled, FOLDER_PATH, OUTPUT_VAL)
vocab = vocab_train.union(vocab_val)

# Write vocabulary
with open(VOCAB_FILE, "w", encoding="utf-8") as vfile:
    for char in sorted(vocab):
        vfile.write(char + '\n')

print(f"[INFO] Processing complete. Vocabulary saved to '{VOCAB_FILE}'")
