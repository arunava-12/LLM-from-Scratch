import os
import lzma
from tqdm import tqdm
import concurrent.futures
import random

def process_file(args):
    directory, filename, output_file = args
    file_path = os.path.join(directory, filename)
    try:
        with lzma.open(file_path, "rt", encoding="utf-8") as infile:
            text = infile.read()
        with open(output_file, "a", encoding="utf-8") as outfile:
            outfile.write(text)
        characters = set(text)
        return characters
    except Exception as e:
        print(f"Error processing {filename}: {e}")
        return set()

def xz_files_in_dir(directory):
    try:
        return [filename for filename in os.listdir(directory)
                if filename.endswith(".xz") and os.path.isfile(os.path.join(directory, filename))]
    except FileNotFoundError:
        print(f"Directory '{directory}' not found.")
        return []

def process_files_in_parallel(files, folder_path, output_file):
    vocab = set()
    if not files:
        print(f"No files to process for output: {output_file}")
        return vocab
    with concurrent.futures.ProcessPoolExecutor(max_workers=4) as executor:
        args = [(folder_path, filename, output_file) for filename in files]
        for characters in tqdm(executor.map(process_file, args), total=len(files)):
            vocab.update(characters)
    return vocab

def safe_sample(files, sample_rate):
    if not files:
        return []
    return random.sample(files, min(len(files), max(1, int(len(files) * sample_rate))))

# ---- Configuration ----
folder_path = "openwebtext"
output_file_train = "output_train.txt"
output_file_val = "output_val.txt"
vocab_file = "vocab.txt"
sample_rate = 0.01

# ---- Check folder exists ----
if not os.path.isdir(folder_path):
    print(f"Directory '{folder_path}' does not exist. Please check the path.")
    exit(1)

# ---- Get all .xz files ----
files = xz_files_in_dir(folder_path)
total_files = len(files)
print(f"Total .xz files found: {total_files}")

if total_files == 0:
    print("No .xz files found in the directory. Exiting.")
    exit(1)

# ---- Train/Val split ----
split_index = int(total_files * 0.9)
files_train = files[:split_index]
files_val = files[split_index:]
print(f"Train files: {len(files_train)}, Val files: {len(files_val)}")

# ---- Sample files safely ----
files_train_sampled = safe_sample(files_train, sample_rate)
files_val_sampled = safe_sample(files_val, sample_rate)
print(f"Sampled Train files: {len(files_train_sampled)}, Sampled Val files: {len(files_val_sampled)}")

# ---- Ensure output files are empty before appending ----
open(output_file_train, 'w').close()
open(output_file_val, 'w').close()

# ---- Process files ----
vocab_train = process_files_in_parallel(files_train_sampled, folder_path, output_file_train)
vocab_val = process_files_in_parallel(files_val_sampled, folder_path, output_file_val)

# ---- Combine and write vocabulary ----
vocab = vocab_train.union(vocab_val)
with open(vocab_file, "w", encoding="utf-8") as vfile:
    for char in sorted(vocab):
        vfile.write(char + '\n')

print("Processing complete.")
