import pandas as pd
import numpy as np

master_matrix = pd.read_csv('mel_spectrogram.csv', header=None).values


sub_files = [
    'submatrix_3.csv',
    'submatrix_3_variant_1.csv',
    'submatrix_3_variant_2.csv',
    'submatrix_3_variant_3.csv',
    'submatrix_3_variant_4.csv'
]

results = []

for file in sub_files:
    sub_matrix = pd.read_csv(file, header=None).values
    s_rows, s_cols = sub_matrix.shape
    m_rows, m_cols = master_matrix.shape
    
    best_col = -1
    min_error = float('inf')
    
    
    for j in range(m_cols - s_cols + 1):
    
        diff = master_matrix[:, j:j+s_cols] - sub_matrix
        error = np.mean(np.square(diff))
        
        if error < min_error:
            min_error = error
            best_col = j
            
    results.append({
        'File': file,
        'Row Range': f"0–{s_rows-1}",
        'Column Range': f"{best_col}–{best_col + s_cols - 1}"
    })

df_results = pd.DataFrame(results)
print(df_results)
