import random
import torch
import numpy as np
import torch.nn as nn
from Levenshtein import distance
from utils.data_handle import write_bits_to_str

BASES = ['A', 'C', 'G', 'T']
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model, device=DEVICE)
        position = torch.arange(0, max_len, dtype=torch.float, device=DEVICE).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, device=DEVICE).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

class TransformerModel(nn.Module):
    def __init__(self, input_dim=4, output_dim=1, d_model=128, nhead=4, 
                 num_layers=3, dim_feedforward=256, dropout=0.1):
        super().__init__()
        self.conv = nn.Conv1d(in_channels=1, out_channels=d_model, 
                              kernel_size=8, stride=8).to(DEVICE)
        self.embedding = nn.Embedding(input_dim, d_model).to(DEVICE)
        self.pos_encoder = PositionalEncoding(d_model, dropout).to(DEVICE)
        encoder_layers = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=num_layers).to(DEVICE)
        self.fc = nn.Linear(d_model, output_dim).to(DEVICE)
        self.d_model = d_model

    def forward(self, src):
        src = src.unsqueeze(1).float()  # (batch_size, 1, seq_length)
        src = self.conv(src)
        src = src.permute(2, 0, 1)
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src)
        output = output.permute(1, 0, 2)
        output = torch.mean(output, dim=1)
        output = self.fc(output)
        return torch.sigmoid(output)

# --------------------------- Error Introduction & Sequence Processing ---------------------------
def introduce_error(sequence, P_sub=0.04, P_del=0.004, P_ins=0.004):
    """Introduce substitution/deletion/insertion errors to DNA sequence and generate labels"""
    seq = list(sequence)
    label = [0] * len(seq)
    max_len = len(seq)
    i = 0

    while i < len(seq):
        r = random.random()
        if r < P_sub:
            # Substitution error (label remains 0)
            original = seq[i]
            new_base = random.choice([b for b in BASES if b != original])
            seq[i] = new_base
            i += 1
            continue

        r = random.random()
        if r < P_del:
            # Deletion error (label is 1)
            del seq[i]
            label = label[:i] + [1] + label[i:]
            if len(seq) < max_len:
                seq.append(random.choice(BASES))
                label.append(0)
            continue

        r = random.random()
        if r < P_ins:
            # Insertion error (label is 1)
            insert_base = random.choice(BASES)
            seq = seq[:i] + [insert_base] + seq[i:]
            label = label[:i] + [1] + label[i:]
            i += 1
        
        i += 1

    # Truncate to original length
    seq = seq[:max_len]
    label = label[:max_len]
    return ''.join(seq), label

def dna_to_ids(seq):
    """Convert DNA sequence to numeric IDs (A=0, C=1, G=2, T=3)"""
    token_map = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
    return [token_map.get(base, 0) for base in seq]

def pad_sequence(ids, max_len):
    """Pad/truncate sequence to specified length"""
    if len(ids) < max_len:
        return ids + [0] * (max_len - len(ids))
    else:
        return ids[:max_len]

# --------------------------- Sequence Correction ---------------------------
def fuzzy_align(str1, str2):
    """Fuzzy align two sequences (handle insertions/deletions)"""
    len1, len2 = len(str1), len(str2)
    alignment = [[0]*(len2+1) for _ in range(len1+1)]
    
    # Initialize boundaries
    for i in range(len1+1):
        alignment[i][0] = i
    for j in range(len2+1):
        alignment[0][j] = j

    # Fill alignment matrix
    for i in range(1, len1+1):
        for j in range(1, len2+1):
            if str1[i-1] == str2[j-1]:
                alignment[i][j] = alignment[i-1][j-1]
            else:
                if i>=2 and j>=2 and str1[i-2]==str2[j-1] and str1[i-1]==str2[j-2]:
                    alignment[i][j] = alignment[i-2][j-2] + 1
                else:
                    alignment[i][j] = min(
                        alignment[i-1][j]+1,  # Deletion
                        alignment[i][j-1]+1,  # Insertion
                        alignment[i-1][j-1]+1 # Substitution
                    )

    # Backtrack to generate alignment results
    aligned_str1, aligned_str2 = '', ''
    i, j = len1, len2
    while i>0 and j>0:
        if str1[i-1] == str2[j-1]:
            aligned_str1 = str1[i-1] + aligned_str1
            aligned_str2 = str2[j-1] + aligned_str2
            i -= 1
            j -= 1
        elif alignment[i][j] == alignment[i-1][j]+1:
            aligned_str1 = str1[i-1] + aligned_str1
            aligned_str2 = '-' + aligned_str2
            i -= 1
        elif alignment[i][j] == alignment[i][j-1]+1:
            aligned_str1 = '-' + aligned_str1
            aligned_str2 = str2[j-1] + aligned_str2
            j -= 1
        else:
            aligned_str1 = str1[i-1] + aligned_str1
            aligned_str2 = str2[j-1] + aligned_str2
            i -= 1
            j -= 1

    # Handle remaining characters
    while i>0:
        aligned_str1 = str1[i-1] + aligned_str1
        aligned_str2 = '-' + aligned_str2
        i -= 1
    while j>0:
        aligned_str1 = '-' + aligned_str1
        aligned_str2 = str2[j-1] + aligned_str2
        j -= 1

    return aligned_str1, aligned_str2

def move_dash_to_right(str1, str2):
    """Move '-' in aligned sequences to the right for better correction"""
    if len(str1) != len(str2):
        raise ValueError("The two sequences must have the same length")
    list1, list2 = list(str1), list(str2)
    swapped = True
    while swapped:
        swapped = False
        for i in range(len(list1)-1):
            # Handle '-' in str1
            if list1[i] == '-' and list2[i] == list1[i+1] and list2[i+1] != '-':
                list1[i], list1[i+1] = list1[i+1], list1[i]
                swapped = True
            # Handle '-' in str2
            if list2[i] == '-' and list1[i] == list2[i+1] and list1[i+1] != '-':
                list2[i], list2[i+1] = list2[i+1], list2[i]
                swapped = True
    return ''.join(list1), ''.join(list2)

def ReturnMinHanminEncodeUnit(linestr, codex, chrEncodeUnitLen=8):
    """Calculate minimum Hamming distance for each 8-bit DNA fragment"""
    hanminglist = []
    possible_units = list(codex.values())
    for index in range(0, len(linestr), chrEncodeUnitLen):
        sub_str = linestr[index:index+chrEncodeUnitLen]
        min_distance = float('inf')
        for unit in possible_units:
            dist = distance(sub_str, unit)
            if dist < min_distance:
                min_distance = dist
        hanminglist.append(min_distance)
    return hanminglist

def IsExceptEncodFrage(tmplist):
    """Determine if encoded fragment needs correction"""
    flag = True
    count = 0
    for vx in tmplist:
        if vx < 2:
            flag = False
        if vx >= 2:
            count += 1
    return (flag is False) and (count >= 2)

def returnPos_Ins_Del(tmplist, obsoletelist, windowsize=3):
    """Find positions that need correction"""
    lenlist = len(tmplist)
    for id in range(lenlist):
        if id in obsoletelist:
            continue
        # Determine window range
        if id + windowsize >= lenlist:
            window = tmplist[id:]
        else:
            window = tmplist[id:id+windowsize]
        # Check if window needs correction
        if IsExceptEncodFrage(window):
            for pos in range(len(window)):
                if window[pos] >= 2:
                    return id + pos
    return -1

def computeRectfiyImpact(line, obsoletelist, totaldict, windowsize=3):
    """Calculate impact range of correction"""
    tmplist = ReturnMinHanminEncodeUnit(line, totaldict)
    id = returnPos_Ins_Del(tmplist, obsoletelist, windowsize)
    return 1000 if id == -1 else id

def retrieveMinHanming(targetstr, totaldict):
    """Get minimum Hamming distance (only for 8-bit fragments)"""
    if len(targetstr) == 8:
        return 8 - totaldict.get(targetstr, (8,))[0]
    return 0

def Ins_Del_CorrectSeq_Strict(linestr, id, obsoletelist, totaldict, 
                              windowsize=3, chrEncodeUnitLen=8):
    """Strict mode correction"""
    rightboudindex = len(linestr) - chrEncodeUnitLen * windowsize
    candiList = []

    if id < rightboudindex:
        start = id * chrEncodeUnitLen
        end = start + chrEncodeUnitLen
        target_fragment = linestr[start:end]
        possible_units = list(totaldict.values())
        
        # Find closest encoding unit
        min_distance = float('inf')
        closest_unit = None
        for unit in possible_units:
            dist = distance(target_fragment, unit)
            if dist < min_distance:
                min_distance = dist
                closest_unit = unit
        
        if closest_unit is not None:
            # Fuzzy align and correct
            aligned_str1, aligned_str2 = fuzzy_align(target_fragment, closest_unit)
            aligned_str1, aligned_str2 = move_dash_to_right(aligned_str1, aligned_str2)
            
            offset = 0
            window = min(8, len(aligned_str1))
            has_deletion, has_insertion = False, False

            for i in range(window):
                a1, a2 = aligned_str1[i], aligned_str2[i]
                if a1 == '-':
                    offset -= 1
                    has_deletion = True
                elif a2 == '-':
                    offset += 1
                    has_insertion = True

            # Construct corrected fragment
            if has_deletion:
                corrected_fragment = aligned_str1[:chrEncodeUnitLen]
            elif has_insertion:
                filtered = [c for c in aligned_str2 if c != '-']
                corrected_fragment = ''.join(filtered[:chrEncodeUnitLen])
            else:
                corrected_fragment = aligned_str2[:chrEncodeUnitLen]

            # Generate new sequence
            start_pos = id * chrEncodeUnitLen
            end_pos = start_pos + chrEncodeUnitLen
            new_line = linestr[:start_pos] + corrected_fragment + linestr[end_pos + offset:]

            # Verify correction effect
            tmplist = ReturnMinHanminEncodeUnit(new_line, totaldict)
            if not IsExceptEncodFrage(tmplist[id:id+windowsize]):
                candiList.append((
                    new_line,
                    computeRectfiyImpact(new_line, obsoletelist, totaldict),
                    retrieveMinHanming(new_line[start_pos:start_pos+chrEncodeUnitLen], totaldict)
                ))

    if candiList:
        return sorted(candiList, key=lambda xt: (xt[1], xt[2]), reverse=True)[0][0]
    return None

def Ins_Del_CorrectSeq_Loose(linestr, id, obsoletelist, totaldict, 
                             windowsize=3, chrEncodeUnitLen=8):
    """Loose mode correction (try inserting/deleting 1-2 bases)"""
    index = id * chrEncodeUnitLen
    rightboudindex = len(linestr) - chrEncodeUnitLen * windowsize
    candiList = []

    # Non-boundary case
    if index < rightboudindex:
        # Insert 1 base
        for vt in range(chrEncodeUnitLen):
            for base in BASES:
                newline = linestr[:index+vt] + base + linestr[index+vt:]
                tmplist = ReturnMinHanminEncodeUnit(newline, totaldict)
                if not IsExceptEncodFrage(tmplist[id:id+windowsize]):
                    candiList.append((newline,
                                      computeRectfiyImpact(newline, obsoletelist, totaldict),
                                      retrieveMinHanming(newline[index:index+chrEncodeUnitLen], totaldict)))
        # Insert 2 bases
        for vt in range(chrEncodeUnitLen):
            for base1 in BASES:
                for base2 in BASES:
                    newline = linestr[:index+vt] + base1 + base2 + linestr[index+vt:]
                    tmplist = ReturnMinHanminEncodeUnit(newline, totaldict)
                    if not IsExceptEncodFrage(tmplist[id:id+windowsize]):
                        candiList.append((newline,
                                          computeRectfiyImpact(newline, obsoletelist, totaldict),
                                          retrieveMinHanming(newline[index:index+chrEncodeUnitLen], totaldict)))
        # Delete 1 base
        for vt in range(chrEncodeUnitLen):
            newline = linestr[:index+vt] + linestr[index+vt+1:]
            tmplist = ReturnMinHanminEncodeUnit(newline, totaldict)
            if not IsExceptEncodFrage(tmplist[id:id+windowsize]):
                candiList.append((newline,
                                  computeRectfiyImpact(newline, obsoletelist, totaldict),
                                  retrieveMinHanming(newline[index:index+chrEncodeUnitLen], totaldict)))
        # Delete 2 bases
        for vt in range(chrEncodeUnitLen):
            newline = linestr[:index+vt] + linestr[index+vt+2:]
            tmplist = ReturnMinHanminEncodeUnit(newline, totaldict)
            if not IsExceptEncodFrage(tmplist[id:id+windowsize]):
                candiList.append((newline,
                                  computeRectfiyImpact(newline, obsoletelist, totaldict),
                                  retrieveMinHanming(newline[index:index+chrEncodeUnitLen], totaldict)))
    # Boundary case
    else:
        termivar = len(linestr) - index
        # Insert 1 base
        if termivar >= 7:
            for vt in range(chrEncodeUnitLen):
                for base in BASES:
                    newline = linestr[:index+vt] + base + linestr[index+vt:]
                    tmplist = ReturnMinHanminEncodeUnit(newline, totaldict)
                    if not IsExceptEncodFrage(tmplist[id:]):
                        candiList.append((newline,
                                          computeRectfiyImpact(newline, obsoletelist, totaldict),
                                          retrieveMinHanming(newline[index:index+chrEncodeUnitLen], totaldict)))
        # Insert 2 bases
        if termivar >= 6:
            for vt in range(chrEncodeUnitLen):
                for base1 in BASES:
                    for base2 in BASES:
                        newline = linestr[:index+vt] + base1 + base2 + linestr[index+vt:]
                        tmplist = ReturnMinHanminEncodeUnit(newline, totaldict)
                        if not IsExceptEncodFrage(tmplist[id:]):
                            candiList.append((newline,
                                              computeRectfiyImpact(newline, obsoletelist, totaldict),
                                              retrieveMinHanming(newline[index:index+chrEncodeUnitLen], totaldict)))
        # Delete 1 base
        if termivar > 9:
            for vt in range(chrEncodeUnitLen):
                newline = linestr[:index+vt] + linestr[index+vt+1:]
                tmplist = ReturnMinHanminEncodeUnit(newline, totaldict)
                if not IsExceptEncodFrage(tmplist[id:]):
                    candiList.append((newline,
                                      computeRectfiyImpact(newline, obsoletelist, totaldict),
                                      retrieveMinHanming(newline[index:index+chrEncodeUnitLen], totaldict)))
        # Delete 2 bases
        if termivar > 10:
            for vt in range(chrEncodeUnitLen):
                newline = linestr[:index+vt] + linestr[index+vt+2:]
                tmplist = ReturnMinHanminEncodeUnit(newline, totaldict)
                if not IsExceptEncodFrage(tmplist[id:]):
                    candiList.append((newline,
                                      computeRectfiyImpact(newline, obsoletelist, totaldict),
                                      retrieveMinHanming(newline[index:index+chrEncodeUnitLen], totaldict)))

    if candiList:
        return sorted(candiList, key=lambda x: (x[1], x[2]), reverse=True)[0][0]
    return None

def correct_sequence(sequence, codex_path='../data/codex.txt', 
                     chrEncodeUnitLen=8, windowsize=3, max_iter=100):
    """
    Main correction function: iteratively detect and correct insertion/deletion errors in DNA sequences
    
    Args:
        sequence: Input DNA sequence
        codex_path: Path to encoding dictionary
        chrEncodeUnitLen: Encoding unit length (default 8)
        windowsize: Detection window size (default 3)
        max_iter: Maximum iteration count (prevent infinite loop)
    
    Returns:
        Corrected DNA sequence (or original sequence if max iterations reached)
    """
    # Load encoding dictionary
    codex = {}
    with open(codex_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip().rstrip(',')
            if line:
                key, value = line.replace("'", "").split(':')
                codex[key] = value

    tmpstr = sequence
    obsoletelist = []
    count = 0

    while count < max_iter:
        tmplist = ReturnMinHanminEncodeUnit(tmpstr, codex)
        id = returnPos_Ins_Del(tmplist, obsoletelist, windowsize)
        if id == -1:
            break  # No positions need correction

        # Try strict correction first, then loose mode if failed
        result = Ins_Del_CorrectSeq_Strict(tmpstr, id, obsoletelist, codex, windowsize, chrEncodeUnitLen)
        if result is None:
            result = Ins_Del_CorrectSeq_Loose(tmpstr, id, obsoletelist, codex, windowsize, chrEncodeUnitLen)

        if result is None:
            obsoletelist.append(id)
        else:
            tmpstr = result
            obsoletelist.append(id)

        count += 1

    # Max iterations reached, return original sequence
    if count == max_iter:
        print(f"Warning: Maximum iteration count {max_iter} reached, returning original sequence")
        return sequence

    return tmpstr

# --------------------------- Decoding & Prediction ---------------------------
def decode(seq, codex, chunk_size=8):
    """
    Decode DNA sequence to binary string
    
    Args:
        seq: DNA sequence
        codex: Encoding dictionary (key:binary, value:DNA)
        chunk_size: Encoding unit length (default 8)
    
    Returns:
        Decoded binary string
    """
    binary_data = []
    reverse_codex = {v: k for k, v in codex.items()}  # Reverse dictionary (DNA→binary)
    i = 0
    while i + chunk_size <= len(seq):
        chunk = seq[i:i+chunk_size]
        decoded = None
        # Match if edit distance ≤ 2
        for dna_code, binary in reverse_codex.items():
            if distance(chunk, dna_code) <= 2:
                decoded = binary
                break
        binary_data.append(decoded if decoded else '00000000')
        i += chunk_size
    return ''.join(binary_data)

def contains_error(seq, codex, windowsize=3):
    """Determine if sequence needs correction"""
    tmplist = ReturnMinHanminEncodeUnit(seq, codex)
    obsoletelist = []
    id = returnPos_Ins_Del(tmplist, obsoletelist, windowsize)
    return id != -1

def predict_sequences(model, sequence_file, max_len=120, output_file='../data/prediction_results.txt',
                      P_sub=0.04, P_del=0.004, P_ins=0.004):
    """
    Predict if DNA sequences contain errors using Transformer model
    
    Args:
        model: Trained Transformer model
        sequence_file: Path to DNA sequence file
        max_len: Maximum sequence length
        output_file: Output path for prediction results
        P_sub/P_del/P_ins: Error introduction probabilities
    
    Returns:
        List of corrupted sequences, list of predicted labels
    """
    # Read sequences
    with open(sequence_file, 'r', encoding='utf-8') as f:
        sequences = [line.strip() for line in f if line.strip()]

    corrupted_sequences = []
    input_ids = []
    true_labels = []

    # Generate corrupted sequences and preprocess
    for seq in sequences:
        corrupted_seq, labels = introduce_error(seq, P_sub, P_del, P_ins)
        corrupted_sequences.append(corrupted_seq)
        label_value = int(sum(labels) > 0)
        true_labels.append(label_value)
        ids = pad_sequence(dna_to_ids(corrupted_seq[:max_len]), max_len)
        input_ids.append(ids)

    # Model prediction
    input_tensor = torch.tensor(input_ids).to(DEVICE)
    model.eval()
    with torch.no_grad():
        outputs = model(input_tensor)
        predictions = (outputs.squeeze() > 0.5).long().cpu().numpy()

    # Statistics calculation
    correct_predictions = 0
    total_predictions = len(sequences)
    false_positive = 0
    false_negative = 0

    # Save results
    with open(output_file, 'w', encoding='utf-8') as out_f:
        out_f.write("Corrupted_Sequence\tTrue_Label\tPredicted_Label\tPrediction_Success\n")
        for i, (seq, true_label, pred) in enumerate(zip(corrupted_sequences, true_labels, predictions)):
            prediction_success = (true_label == pred)
            if prediction_success:
                correct_predictions += 1
            else:
                if true_label == 0 and pred == 1:
                    false_positive += 1
                elif true_label == 1 and pred == 0:
                    false_negative += 1
            out_f.write(f"{seq}\t{true_label}\t{pred}\t{prediction_success}\n")
            print(f"Sequence {i+1}: True_Label={true_label} | Predicted_Label={pred} | Correct={prediction_success}")

    # Calculate metrics
    accuracy = correct_predictions / total_predictions
    false_positive_rate = false_positive / total_predictions
    false_negative_rate = false_negative / total_predictions
    print(f"\n✅ Prediction Accuracy: {accuracy:.4f}")
    print(f"❌ False Positive Rate (0→1): {false_positive_rate:.4f}")
    print(f"❌ False Negative Rate (1→0): {false_negative_rate:.4f}")

    return corrupted_sequences, predictions

# --------------------------- Main Function ---------------------------
if __name__ == "__main__":
    # 1. Load encoding dictionary
    CODEX_PATH = '../data/codex.txt'
    codex = {}
    with open(CODEX_PATH, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip().rstrip(',')
            if line:
                key, value = line.replace("'", "").split(':')
                codex[key] = value

    # 2. Load model
    model = TransformerModel().to(DEVICE)
    model.load_state_dict(torch.load('../models/best_model_main.pth', map_location=DEVICE))

    # 3. Predict error sequences
    corrupted_sequences, predictions = predict_sequences(
        model,
        '../data/pome_dna.txt',
        max_len=120,
        output_file='../data/prediction_results.txt',
        P_sub=0.04, P_del=0.004, P_ins=0.004
    )

    # 4. Decode (correction + decoding)
    reverse_codex = {v: k for k, v in codex.items()}
    all_decoded_bits = []
    for idx, (seq, label) in enumerate(zip(corrupted_sequences, predictions)):
        needs_correction = label == 1 
        if needs_correction:
            print(f"[{idx+1}/{len(corrupted_sequences)}] Sequence needs correction")
            corrected_sequence = correct_sequence(seq, CODEX_PATH)
            decoded = decode(corrected_sequence, reverse_codex)
        else:
            print(f"[{idx+1}/{len(corrupted_sequences)}] Sequence has no errors")
            decoded = decode(seq, reverse_codex)
        all_decoded_bits.append(decoded)

    # 5. Restore to text
    recovered_text = write_bits_to_str(all_decoded_bits, bit_size=None, need_log=True)
    with open("../data/recovered_Text1.txt", "w", encoding="utf8") as f:
        f.write(recovered_text)
    print("\n✅ Decoding completed, recovered text saved to data/recovered_Text1.txt")