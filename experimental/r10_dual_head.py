import argparse
import pandas as pd

def parse_args():
    parser = argparse.ArgumentParser(description="Avenue A: R10 Dual-Head Extractor")
    parser.add_argument("--fastq", required=True, help="Path to R10 FASTQ file")
    parser.add_argument("--eventalign", required=True, help="Path to R10 eventalign.txt file")
    parser.add_argument("--target_base", default="T", help="Target base to center on")
    parser.add_argument("--output", default="r10_features.csv", help="Output 15-feature CSV")
    return parser.parse_args()

def main():
    args = parse_args()
    
    print("Mapping R10 FASTQ Read Indices to UUIDs...")
    index_to_id = {}
    with open(args.fastq, "r") as f:
        for i, line in enumerate(f):
            if i % 4 == 0:
                rid = line.strip().split()[0][1:]
                index_to_id[i // 4] = rid

    print("Parsing EventAlign for 5-kmer Dual-Head Context Window...")
    features = []
    read_ids = []
    current_read = None
    event_buffer = [] 
    
    with open(args.eventalign, "r") as f:
        next(f) 
        for line in f:
            parts = line.strip().split()
            if len(parts) < 15: continue
            
            kmer = parts[2]
            read_idx = int(parts[3])
            event_mean = float(parts[6])
            event_stdv = float(parts[7])
            event_length = float(parts[8])
            
            if read_idx not in index_to_id: continue
            rid = index_to_id[read_idx]
            
            if rid != current_read:
                current_read = rid
                event_buffer = []
                
            event_buffer.append({
                'kmer': kmer, 'mean': event_mean, 'stdv': event_stdv, 'length': event_length
            })
            
            # R10 requires a 5-event window to capture both reader heads
            if len(event_buffer) == 5:
                # The target base is exactly in the center (index 2)
                mid_base = event_buffer[2]['kmer'][2]
                
                if mid_base == args.target_base:
                    row_features = []
                    # Flatten all 5 events into a 15-dimensional vector
                    for i in range(5):
                        row_features.extend([
                            event_buffer[i]['mean'], 
                            event_buffer[i]['stdv'], 
                            event_buffer[i]['length']
                        ])
                    
                    features.append(row_features)
                    read_ids.append(rid)
                
                # Slide window forward
                event_buffer.pop(0)

    print(f"Saving 15-dimensional R10 feature vectors...")
    
    # Generate column names for the 5 positions
    cols = []
    positions = ["Approach", "Head_1", "Center", "Head_2", "Exit"]
    for pos in positions:
        cols.extend([f"{pos}_Mean", f"{pos}_Stdv", f"{pos}_Dwell"])
        
    df = pd.DataFrame(features, columns=cols)
    df.insert(0, "Read_ID", read_ids)
    df.to_csv(args.output, index=False)
    
    print(f"R10 Extraction Complete! Extracted {len(features)} sequences to {args.output}")

if __name__ == "__main__":
    main()
