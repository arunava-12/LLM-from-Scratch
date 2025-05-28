import os
import lzma
from tqdm import tqdm
from multiprocessing import cpu_count
import concurrent.futures

def process_file(args):
    directory, filename, temp_output_file = args
    file_path = os.path.join(directory, filename)
    
    try:
        with lzma.open(file_path, "rt", encoding="utf-8") as infile:
            text = infile.read()
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return set()
    
    with open(temp_output_file, "w", encoding="utf-8") as outfile:
        outfile.write(text)
    
    return set(text)

def xz_files_in_dir(directory):
    try:
        all_files = os.listdir(directory)
    except Exception as e:
        print(f"Error listing directory {directory}: {e}")
        return []
    files = [f for f in all_files if f.endswith(".xz") and os.path.isfile(os.path.join(directory, f))]
    return files

def process_files_in_parallel(files, folder_path, output_file):
    vocab = set()
    temp_dir = output_file + "_temp"
    os.makedirs(temp_dir, exist_ok=True)
    
    args = []
    for filename in files:
        temp_output_file = os.path.join(temp_dir, filename + ".txt")
        args.append((folder_path, filename, temp_output_file))
    
    with concurrent.futures.ProcessPoolExecutor(max_workers=cpu_count()) as executor:
        for characters in tqdm(executor.map(process_file, args), total=len(files), desc=f"Processing {output_file}"):
            vocab.update(characters)
    
    # Merge temp files into one output file
    with open(output_file, "w", encoding="utf-8") as outfile:
        for filename in files:
            temp_output_file = os.path.join(temp_dir, filename + ".txt")
            try:
                with open(temp_output_file, "r", encoding="utf-8") as infile:
                    outfile.write(infile.read())
            except Exception as e:
                print(f"Error reading temp file {temp_output_file}: {e}")
    
    # Clean up temp files and folder
    for f in os.listdir(temp_dir):
        os.remove(os.path.join(temp_dir, f))
    os.rmdir(temp_dir)
    
    return vocab

if __name__ == "__main__":
    folder_path = "openwebtext"
    
    print(f"Current working directory: {os.getcwd()}")
    
    files = xz_files_in_dir(folder_path)
    print(f"Found {len(files)} .xz files in '{folder_path}':")
    print(files)
    
    if not files:
        print("No .xz files found. Please check folder path and file extensions.")
        exit(1)
    
    total_files = len(files)
    split_index = int(total_files * 0.9)
    files_train = files[:split_index]
    files_val = files[split_index:]
    
    output_file_train = "output_train.txt"
    output_file_val = "output_val.txt"
    vocab_file = "vocab.txt"
    
    print(f"Processing {len(files_train)} training files...")
    vocab_train = process_files_in_parallel(files_train, folder_path, output_file_train)
    
    print(f"Processing {len(files_val)} validation files...")
    vocab_val = process_files_in_parallel(files_val, folder_path, output_file_val)
    
    vocab = vocab_train.union(vocab_val)
    
    print(f"Writing vocabulary file '{vocab_file}' with {len(vocab)} unique characters.")
    with open(vocab_file, "w", encoding="utf-8") as vfile:
        for char in sorted(vocab):
            vfile.write(char + '\n')
    
    print("Done.")
