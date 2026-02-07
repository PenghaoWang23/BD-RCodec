import random
from collections import Counter
from Levenshtein import distance as edit_distance
from utils.data_handle import read_bits_from_str, write_dna_file

BASES = ['A', 'T', 'G', 'C']

def gc_content(seq):
    """Calculate GC content"""
    gc = seq.count('G') + seq.count('C')
    return gc / len(seq) if len(seq) > 0 else 0

def max_homopolymer_run(seq):
    """Calculate maximum homopolymer length (consecutive identical bases)"""
    max_run = run = 1
    for i in range(1, len(seq)):
        if seq[i] == seq[i - 1]:
            run += 1
            max_run = max(max_run, run)
        else:
            run = 1
    return max_run

def has_tandem_repeat(seq, max_repeat_len):
    """Detect presence of consecutive repeated substrings"""
    for l in range(1, max_repeat_len + 1):
        for i in range(len(seq) - 2 * l + 1):
            repeat_unit = seq[i:i + l]
            next_unit = seq[i + l:i + 2 * l]
            if repeat_unit == next_unit:
                return True
    return False

def generate_valid_dna_sequences(count, length, gc_range=(0.4, 0.6),
                                  max_homopolymer=4, min_edit_distance=3,
                                  max_repeat_len=2):
    """Generate DNA coding units that meet constraints"""
    dna_set = []
    attempts = 0
    max_attempts = count * 100

    while len(dna_set) < count and attempts < max_attempts:
        candidate = ''.join(random.choices(BASES, k=length))
        attempts += 1
        
        # Filter sequences with unsatisfactory GC content
        if not (gc_range[0] <= gc_content(candidate) <= gc_range[1]):
            continue
        # Filter sequences with overly long homopolymers
        if max_homopolymer_run(candidate) > max_homopolymer:
            continue
        # Filter sequences with repeated substrings
        if has_tandem_repeat(candidate, max_repeat_len):
            continue
        # Filter sequences with insufficient edit distance
        if all(edit_distance(candidate, existing) >= min_edit_distance for existing in dna_set):
            dna_set.append(candidate)

    if len(dna_set) < count:
        raise ValueError(f"Failed to generate {count} valid DNA coding units (exceeded maximum attempts).")

    return dna_set

def read_text_from_file(filepath):
    """Read text file"""
    with open(filepath, 'r', encoding='utf-8') as f:
        return f.read()

def process_file_and_generate_codex(text_file, codex_file, dna_output_file):
    """
    Main function: Read text → Generate coding dictionary → Convert text to DNA sequences
    
    Args:
        text_file: Path to input text file
        codex_file: Path to output coding dictionary
        dna_output_file: Path to output DNA sequence file
    """
    # 1. Read text and convert to binary matrix
    text = read_text_from_file(text_file)
    matrix, size = read_bits_from_str(text, segment_length=120, need_log=False)

    # 2. Flatten to complete binary string
    binary_string = ''.join([''.join(map(str, row)) for row in matrix])

    # 3. Count 8-bit binary combinations
    group_counts = Counter(binary_string[i:i+8] for i in range(0, len(binary_string), 8))
    unique_binaries = list(group_counts.keys())
    print(f"Found {len(unique_binaries)} unique 8-bit binary combinations.")

    # 4. Generate DNA coding units
    dna_list = generate_valid_dna_sequences(
        count=len(unique_binaries),
        length=8,
        gc_range=(0.4, 0.6),
        max_homopolymer=4,
        min_edit_distance=3,
        max_repeat_len=2,
    )

    # 5. Save coding dictionary
    codex = dict(zip(unique_binaries, dna_list))
    with open(codex_file, 'w', encoding='utf-8') as f:
        for binary, dna in codex.items():
            f.write(f"'{binary}':'{dna}',\n")
    print(f"Coding dictionary saved to {codex_file}")

    # 6. Encode to DNA sequences
    encoded_sequences = []
    for row in matrix:
        dna_row = []
        for i in range(0, len(row), 8):
            bits = ''.join(map(str, row[i:i+8]))
            dna_row.append(codex[bits])
        encoded_sequences.append(''.join(dna_row))

    # 7. Save DNA sequences
    write_dna_file(dna_output_file, encoded_sequences, need_log=False)
    print(f"DNA encoding file written to: {dna_output_file}")

if __name__ == '__main__':
    # Example: Modify paths according to your actual situation
    TEXT_FILE = '../data/input_text.txt'
    CODEX_FILE = '../data/codex.txt'
    DNA_OUTPUT_FILE = '../data/pome_dna.txt'

    process_file_and_generate_codex(TEXT_FILE, CODEX_FILE, DNA_OUTPUT_FILE)