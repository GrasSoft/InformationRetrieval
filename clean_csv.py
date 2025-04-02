import pandas as pd
import re
import os 
import sys


def clean_text(value):
    if isinstance(value, str):
        # Remove all special characters and punctuation, but keep letters, numbers, and normal spaces
        cleaned = re.sub(r'[^A-Za-z0-9\s]', '', value)
        
        # Replace newlines, tabs, and multiple spaces with a single space
        cleaned = re.sub(r'\s+', ' ', cleaned)
        
        # Final cleanup
        cleaned = cleaned.strip()

        # Replace empty or whitespace-only strings with NaN
        return cleaned if cleaned else np.nan
    return value

def main():
    for filename in os.listdir(f"./query_short/{sys.argv[1]}"):
        csv_file = filename  

        # Read the CSV file
        df = pd.read_csv(f"./query_short/{sys.argv[1]}/{csv_file}")

        df['query'] = df['query'].apply(clean_text)

        # Overwrite the original file
        df.to_csv(f"./query_short/{sys.argv[1]}/clean_{csv_file}", index=False)

        print(f"Cleaned and saved in-place to {csv_file}")

if __name__ == '__main__':
    main()