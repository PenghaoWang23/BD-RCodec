import random
import torch
import torch.optim as optim
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from collections import Counter

# Global configuration
BASES = ['A', 'C', 'G', 'T']
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Coding dictionary (consistent with encode.py)
CODEX = {
    '00100000':'CGATACGA', '01010111':'GTACGTCT', '01101001':'CGCAGTAT',
    '01101100':'TCATCTGC', '01100001':'GCAGATGT', '01101101':'GCGTAGAT',
    '01010011':'GTGACTGA', '01101000':'GATAGACG', '01101011':'TACGCATC',
    '01100101':'TCTACGCT', '01110011':'GTCTGCTA', '01110000':'AGCGATCT',
    '01110010':'TCGAGTAG', '00101110':'ATGCTCAC', '01101111':'CAGTCAGA',
    '01101110':'TCTGCTGT', '01110100':'ATGACGAC', '00001010':'AGCACTGT',
    '01001001':'ACTCGATG', '01000110':'CTACTCGT', '01010010':'CTAGTGAC',
    '01001111':'CATGTACG', '01001101':'CAGACGAT', '01100110':'TACAGTGC',
    '01100011':'TGTCAGTG', '01110101':'ACGACATG', '01110111':'CGTGATAC',
    '01100100':'GCTGACAT', '00101100':'TCGTCGTA', '01010100':'TACGAGCA',
    '01100010':'CTGTCTAG', '01111001':'ATCTGAGC', '00100111':'CACTGAGT',
    '01100111':'TGATGAGC', '01110110':'ACAGTCTG', '01000010':'ATAGCGTC',
    '01001000':'GTCTATCG', '00111010':'TATGCGTG', '00101101':'CTATGCAG',
    '01000001':'CTCACGTA', '01010000':'AGCTAGCA', '01111010':'AGTGCATC',
    '01111000':'GATGCTAG', '00100001':'GTCAGACT', '01001100':'TAGCTCAG',
    '01001110':'CGACTGTA', '00111011':'TGAGCGAT', '01000100':'CAGATGCA',
    '00111111':'ATCGACTC', '01000011':'TGCGTCAT', '01010110':'ATCGTGCA',
    '01010101':'ACAGCAGT', '01110001':'GCACTATC', '01101010':'CAGCATAC',
    '01011000':'GTGATCAG', '01000111':'GCTATGTC', '01011001':'GACGATAC',
    '01000101':'AGCATCGA', '01001011':'CTCATACG', '01001010':'CTGAGCTA',
    '00001001':'CGTAGACA', '01011011':'GTAGCTAC', '01011101':'AGCTGTAC',
    '00000000':'TATGTCGC'
}

# --------------------------- Data Generation ---------------------------
def introduce_error(sequence, P_sub=0.02, P_del=0.002, P_ins=0.002):
    """Introduce errors to sequence (for training)"""
    seq = list(sequence)
    label = [0] * len(seq)
    max_len = len(seq)
    i = 0

    while i < len(seq):
        r = random.random()
        if r < P_sub:
            original = seq[i]
            new_base = random.choice([b for b in BASES if b != original])
            seq[i] = new_base
            i += 1
            continue

        r = random.random()
        if r < P_del:
            del seq[i]
            label = label[:i] + [1] + label[i:]
            if len(seq) < max_len:
                seq.append(random.choice(BASES))
                label.append(0)
            continue

        r = random.random()
        if r < P_ins:
            insert_base = random.choice(BASES)
            seq = seq[:i] + [insert_base] + seq[i:]
            label = label[:i] + [1] + label[i:]
            i += 1
        
        i += 1

    seq = seq[:max_len]
    label = label[:max_len]
    return ''.join(seq), label

def build_dataset(num_samples=10000, units_per_seq=15):
    """Build training dataset"""
    data = []
    for _ in range(num_samples):
        dna_seq = []
        for _ in range(units_per_seq):
            _, dna = random.choice(list(CODEX.items()))
            dna_seq.append(dna)
        full_dna = ''.join(dna_seq)
        error_dna, label = introduce_error(full_dna)
        data.append((full_dna, error_dna, label))
    return data

def dna_to_ids(seq):
    """Convert DNA to numeric IDs"""
    base_map = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
    return [base_map[b] for b in seq]

def pad_sequence(seq, max_len, pad_value=0):
    """Pad sequence to fixed length"""
    return seq + [pad_value] * (max_len - len(seq))

def preprocess_dataset(data, max_len):
    """Preprocess dataset"""
    input_ids, labels = [], []
    for _, dna_with_error, label in data:
        ids = dna_to_ids(dna_with_error[:max_len])
        ids = pad_sequence(ids, max_len)
        label_value = int(sum(label[:max_len]) > 0)
        input_ids.append(ids)
        labels.append(label_value)
    return input_ids, labels

# --------------------------- Model Definition ---------------------------
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
        src = src.unsqueeze(1).float()
        src = self.conv(src)
        src = src.permute(2, 0, 1)
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src)
        output = output.permute(1, 0, 2)
        output = torch.mean(output, dim=1)
        output = self.fc(output)
        return torch.sigmoid(output)

# --------------------------- Training & Evaluation ---------------------------
def train(model, iterator, optimizer, criterion, device):
    """Training function"""
    model.train()
    for batch in iterator:
        optimizer.zero_grad()
        X, y = batch
        X, y = X.to(device), y.to(device)
        predictions = model(X)
        loss = criterion(predictions.squeeze(), y.float())
        loss.backward()
        optimizer.step()

def evaluate(model, iterator, criterion, device):
    """Evaluation function"""
    model.eval()
    epoch_loss = 0
    total_correct = 0
    total_samples = 0

    with torch.no_grad():
        for batch in iterator:
            X, y = batch
            X, y = X.to(device), y.to(device)

            predictions = model(X)
            loss = criterion(predictions.squeeze(), y.float())
            epoch_loss += loss.item()

            # Calculate accuracy
            predicted_classes = (predictions > 0.5).float()
            correct = (predicted_classes.squeeze() == y).sum().item()
            total_correct += correct
            total_samples += y.size(0)

    avg_loss = epoch_loss / len(iterator)
    accuracy = total_correct / total_samples
    return avg_loss, accuracy

# --------------------------- Main Training Pipeline ---------------------------
if __name__ == "__main__":
    # 1. Configuration parameters
    UNITS_PER_SEQ = 15
    MAX_LEN = UNITS_PER_SEQ * 8
    NUM_SAMPLES = 10000
    BATCH_SIZE = 64
    NUM_EPOCHS = 100
    LEARNING_RATE = 0.001

    # 2. Build dataset
    print("Building dataset...")
    dataset = build_dataset(NUM_SAMPLES, UNITS_PER_SEQ)
    input_ids, labels = preprocess_dataset(dataset, max_len=MAX_LEN)

    # 3. Split train/test sets
    input_tensor = torch.tensor(input_ids)
    label_tensor = torch.tensor(labels)
    train_data = TensorDataset(input_tensor[:8000], label_tensor[:8000])
    test_data = TensorDataset(input_tensor[8000:], label_tensor[8000:])
    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False)

    # 4. Initialize model
    print("Initializing model...")
    model = TransformerModel().to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.BCELoss()

    # 5. Train model
    print("Starting training...")
    train_losses, train_accuracies = [], []
    test_losses, test_accuracies = [], []
    best_accuracy = 0.0

    for epoch in range(NUM_EPOCHS):
        train(model, train_loader, optimizer, criterion, DEVICE)
        
        # Evaluation
        train_loss, train_acc = evaluate(model, train_loader, criterion, DEVICE)
        test_loss, test_acc = evaluate(model, test_loader, criterion, DEVICE)
        
        # Save best model
        if test_acc > best_accuracy:
            best_accuracy = test_acc
            torch.save(model.state_dict(), '../models/best_model_main.pth')
            print(f"✅ Saving best model (accuracy: {test_acc:.4f})")
        
        # Record metrics
        train_losses.append(train_loss)
        train_accuracies.append(train_acc)
        test_losses.append(test_loss)
        test_accuracies.append(test_acc)
        
        print(f"Epoch {epoch+1}/{NUM_EPOCHS}: "
              f"Train Loss={train_loss:.4f}, Train Accuracy={train_acc:.4f}, "
              f"Test Loss={test_loss:.4f}, Test Accuracy={test_acc:.4f}")