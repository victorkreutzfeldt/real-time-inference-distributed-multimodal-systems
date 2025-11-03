import pandas as pd
import math
import os
import json
from collections import defaultdict, Counter

print("="*10)
print("Generating dataset annotation file!")
print("="*10)
print("\n")

# = Config =
ANNOTATION_FILE = 'data/Annotations.txt'
TRAIN_FILE = 'data/trainSet.txt'
VAL_FILE = 'data/valSet.txt'
TEST_FILE = 'data/testSet.txt'
TOKEN_DURATION = 1.0  # seconds per token
NUM_TOTAL_TOKENS = 10  # total tokens per video, e.g. 10 seconds duration
OUTPUT_CSV = 'data/annotations.csv'

# = Helper to load split file =
def load_split_file(file_path):
    split = set()
    with open(file_path, 'r') as f:
        for line in f:
            parts = [p.strip() for p in line.strip().split('&')]
            if len(parts) >= 2:
                video_id = parts[1]
                split.add(video_id)
    return split

train_set = load_split_file(TRAIN_FILE)
val_set = load_split_file(VAL_FILE)
test_set = load_split_file(TEST_FILE)

def get_split(video_id):
    if video_id in train_set:
        return 'train'
    elif video_id in val_set:
        return 'val'
    elif video_id in test_set:
        return 'test'
    else:
        return 'unknown'

# = Parse annotation file to collect labels per token =
token_labels = defaultdict(lambda: defaultdict(set))  # video_id -> token_index -> set(labels)
label_counter = Counter()

with open(ANNOTATION_FILE, 'r') as f:
    next(f)  # Skip header line
    for line in f:
        parts = [p.strip() for p in line.strip().split('&')]
        if len(parts) != 5:
            continue
        
        label, video_id, _, start_str, end_str = parts
        label = label.lower()
        start_time = float(start_str)
        end_time = float(end_str)
        split = get_split(video_id)
        if split == 'unknown':
            continue
        
        start_token = int(start_time // TOKEN_DURATION)
        end_token = int(math.ceil(end_time / TOKEN_DURATION))
        for i in range(start_token, end_token):
            if i >= NUM_TOTAL_TOKENS:
                continue  # overflow tokens ignored
            token_labels[video_id][i].add(label)
            label_counter[label] += 1

# Add background label if not exists
if 'background' not in label_counter:
    label_counter['background'] = 0

# Sort alphabetically by label name
sorted_labels = sorted(label_counter.keys())

# Create mappings
label_to_index = {label: idx for idx, label in enumerate(sorted_labels)}
index_to_label = {idx: label for label, idx in label_to_index.items()}

# Fill in missing tokens and generate rows
rows = []
all_video_ids = train_set | val_set | test_set
for video_id in all_video_ids:
    split = get_split(video_id)
    for i in range(NUM_TOTAL_TOKENS):
        token_start = i * TOKEN_DURATION
        token_end = token_start + TOKEN_DURATION
        labels = token_labels[video_id].get(i, {'background'})

        # Update background count if empty
        if labels == {'background'}:
            label_counter['background'] += 1
        
        label_idxs = []
        for label in labels:
            label_idxs.append(label_to_index[label])
    
        rows.append({
            'video_id': video_id,
            'token_idx': i,
            'start': round(token_start, 3),
            'end': round(token_end, 3),
            'labels': sorted(labels),
            'labels_idx': sorted(label_idxs),
            'split': split
        })

# Save to CSV
df = pd.DataFrame(rows)
df = df.sort_values(by=['video_id', 'token_idx']).reset_index(drop=True)
df.to_csv(OUTPUT_CSV, index=False)
print(f"Saved annotation CSV to {OUTPUT_CSV}")

# === Save mappings to JSON ===
MAPPINGS_DIR = os.path.dirname(OUTPUT_CSV) or "."
LABEL2IDX_PATH = os.path.join(MAPPINGS_DIR, "label_to_index.json")
IDX2LABEL_PATH = os.path.join(MAPPINGS_DIR, "index_to_label.json")

with open(LABEL2IDX_PATH, "w") as f:
    json.dump(label_to_index, f, ensure_ascii=False, indent=2, sort_keys=True)

# Convert keys to string for JSON
idx_to_label_str_keys = {str(k): v for k, v in index_to_label.items()}
with open(IDX2LABEL_PATH, "w") as f:
    json.dump(idx_to_label_str_keys, f, ensure_ascii=False, indent=2, sort_keys=True)

print(f"Saved label_to_index to {LABEL2IDX_PATH}")
print(f"Saved index_to_label to {IDX2LABEL_PATH}")

# = Reporting: Video counts and label counts per split =
print("\n=== Dataset Summary ===\n")

# Number of videos per split
video_counts = {
    'train': len(train_set),
    'val': len(val_set),
    'test': len(test_set),
    'total': len(all_video_ids)
}
print("Number of videos per split:")
for split_name, count in video_counts.items():
    print(f"  {split_name:5}: {count}")

# Number of tokens per split
token_counts_split = df.groupby('split').size()
total_tokens = len(df)
print(f"\nTotal tokens (all splits): {total_tokens}")
print("Tokens per split:")
for split_name in ['train', 'val', 'test']:
    count = token_counts_split.get(split_name, 0)
    pct = 100.0 * count / total_tokens if total_tokens > 0 else 0.0
    print(f"  {split_name:5}: {count} ({pct:.2f}%)")

# Label counts overall and per split (including background)
print("\nLabel distributions (all splits):")
all_labels_flat = [lbl for labels in df['labels'] for lbl in labels]
label_counts_overall = Counter(all_labels_flat)
total_labels = sum(label_counts_overall.values())
for label, count in sorted(label_counts_overall.items()):
    pct = 100.0 * count / total_labels if total_labels > 0 else 0.0
    print(f"  {label:15}: {count} ({pct:.2f}%)")

# Label counts per split
print("\nLabel distributions by split:")
for split_name in ['train', 'val', 'test']:
    split_df = df[df['split'] == split_name]
    split_labels_flat = [lbl for labels in split_df['labels'] for lbl in labels]
    split_label_counts = Counter(split_labels_flat)
    split_total = sum(split_label_counts.values())
    print(f"\nSplit: {split_name} (tokens: {len(split_df)})")
    for label, count in sorted(split_label_counts.items()):
        pct = 100.0 * count / split_total if split_total > 0 else 0.0
        print(f"  {label:15}: {count} ({pct:.2f}%)")
