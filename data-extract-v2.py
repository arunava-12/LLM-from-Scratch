import os
import lzma
from tqdm import tqdm
from multiprocessing import cpu_count
from concurrent.futures import ProcessPoolExecutor

def extract_text_from_xz(file_path):
    try:
        with lzma.open(file_path, "rt", encoding="utf-8") as infile:
            return infile.read()
    except Exception as e:
        print(f"[ERROR] Failed to read {file_path}: {e}")
        return None

def process_file(args):
    directory, filename, temp_output_path = args
    file_path = os.path.join(directory, filename)
    text = extract_text_from_xz(file_path)

    if text is None:
        return set()

    try:
        with open(temp_output_path, "w", encoding="utf-8") as outfile:
            outfile.write(text)
    except Exception as e:
        print(f"[ERROR] Failed to write {temp_output_path}: {e}")
        return set()

    return set(text)

def list_xz_files(directory):
    try:
        return sorted([
            f for f in os.listdir(directory)
            if f.endswith(".xz") and os.path.isfile(os.path.join(directory, f))
        ])
    except Exception as e:
        print(f"[ERROR] Failed to list files in {directory}: {e}")
        return []

def process_files(files, input_dir, output_file):
    vocab = set()
    temp_dir = output_file + "_tmp"
    os.makedirs(temp_dir, exist_ok=True)

    args = [
        (input_dir, filename, os.path.join(temp_dir, filename + ".txt"))
        for filename in files
    ]

    with ProcessPoolExecutor(max_workers=cpu_count()) as executor:
        for char_set in tqdm(executor.map(process_file, args), total=len(files), desc=f"Processing {output_file}"):
            vocab.update(char_set)

    with open(output_file, "w", encoding="utf-8") as outfile:
        for _, _, temp_output_path in args:
            try:
                with open(temp_output_path, "r", encoding="utf-8") as infile:
                    outfile.write(infile.read())
            except Exception as e:
                print(f"[ERROR] Failed to merge {temp_output_path}: {e}")

    for f in os.listdir(temp_dir):
        os.remove(os.path.join(temp_dir, f))
    os.rmdir(temp_dir)

    return vocab

def write_vocab_file(vocab, output_path):
    try:
        with open(output_path, "w", encoding="utf-8") as vfile:
            for char in sorted(vocab):
                vfile.write(char + "\n")
    except Exception as e:
        print(f"[ERROR] Could not write vocab file {output_path}: {e}")

def main():
    folder_path = "openwebtext"
    output_train = "output_train.txt"
    output_val = "output_val.txt"
    vocab_output = "vocab.txt"

    print(f"[INFO] Scanning directory: {folder_path}")
    files = list_xz_files(folder_path)

    if not files:
        print("[ERROR] No .xz files found.")
        return

    print(f"[INFO] Found {len(files)} .xz files.")

    split_idx = int(0.9 * len(files))
    files_train, files_val = files[:split_idx], files[split_idx:]

    print(f"[INFO] Processing {len(files_train)} training files...")
    vocab_train = process_files(files_train, folder_path, output_train)

    print(f"[INFO] Processing {len(files_val)} validation files...")
    vocab_val = process_files(files_val, folder_path, output_val)

    combined_vocab = vocab_train.union(vocab_val)
    print(f"[INFO] Writing combined vocabulary of {len(combined_vocab)} unique characters.")
    write_vocab_file(combined_vocab, vocab_output)

    print("[INFO] Processing complete.")

if __name__ == "__main__":
    main()
