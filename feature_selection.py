import pandas as pd
import string

def clean_text(text):
    """
    Fungsi untuk membersihkan teks tweet (TANPA REGEX).
    Hanya melakukan lowercasing dan merapikan spasi.
    """
    if pd.isna(text) or text == '':
        return ''
    
    # Convert to string and lowercase
    text = str(text).lower()
    
    # --- BAGIAN REGEX DIHAPUS ---
    # URL, Mention, Hashtag, RT, Angka, dan Tanda Baca sekarang AKAN TETAP ADA.
    
    # Remove extra whitespace using standard string methods (bukan regex)
    # Ini akan mengubah spasi ganda/tab/enter menjadi 1 spasi
    text = " ".join(text.split())
    
    return text

def preprocess_data(input_file, output_file):
    """
    Fungsi utama untuk preprocessing data
    """
    print("🔄 Memuat data...")
    # Load data
    df = pd.read_csv(input_file)
    
    # Simpan jumlah awal untuk statistik
    initial_total = len(df)
    print(f"📊 Total data awal: {initial_total} tweets")
    
    # Apply text cleaning
    print("🧹 Membersihkan teks (Basic: Lowercase & Spasi)...")
    df['cleaned_text'] = df['full_text'].apply(clean_text)
    
    # Remove empty texts after cleaning
    df = df[df['cleaned_text'].str.len() > 0].copy()
    
    # --- HAPUS DUPLIKAT ---
    print("🔍 Menghapus duplikasi berdasarkan 'cleaned_text'...")
    count_before_dedupe = len(df)
    
    # Hapus duplikat
    df.drop_duplicates(subset=['cleaned_text'], keep='first', inplace=True)
    
    duplicates_removed = count_before_dedupe - len(df)
    print(f"🗑️ Duplikasi dihapus: {duplicates_removed} tweets")
    
    print(f"✅ Data setelah cleaning & dedupe: {len(df)} tweets")
    
    # Add text statistics
    print("📈 Menambahkan statistik teks...")
    df['text_length'] = df['cleaned_text'].str.len()
    df['word_count'] = df['cleaned_text'].str.split().str.len()
    
    # Add binary flag for image
    if 'image_url' in df.columns:
        df['has_image'] = df['image_url'].notna().astype(int)
    else:
        df['has_image'] = 0
    
    # Convert engagement metrics to numeric
    for col in ['favorite_count', 'retweet_count', 'reply_count']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).astype(int)
    
    # Select relevant columns for output
    output_columns = [
        'cleaned_text'
    ]
    
    final_columns = [col for col in output_columns if col in df.columns]
    df_output = df[final_columns].copy()
    
    # Save to CSV
    print(f"💾 Menyimpan hasil ke {output_file}...")
    df_output.to_csv(output_file, index=False, encoding='utf-8')
    
    # Print statistics
    print("\n" + "="*50)
    print("📊 STATISTIK PREPROCESSING")
    print("="*50)
    print(f"Total tweets awal        : {initial_total}")
    print(f"Total tweets akhir       : {len(df_output)}")
    print(f"Tweets dihapus (kosong)  : {initial_total - count_before_dedupe}")
    print(f"Tweets dihapus (duplikat): {duplicates_removed}")
    print(f"Total tweets dihapus     : {initial_total - len(df_output)}")
    print("="*50)
    
    # Menggunakan df (bukan df_output) karena kolom statistik ada di df
    print(f"Rata-rata panjang teks   : {df['text_length'].mean():.2f} karakter")
    print(f"Rata-rata jumlah kata    : {df['word_count'].mean():.2f} kata")
    
    if 'has_image' in df.columns:
        print(f"Tweets dengan gambar     : {df['has_image'].sum()}")
        print(f"Tweets tanpa gambar      : {len(df) - df['has_image'].sum()}")
    print("="*50)
    
    # Show sample
    print("\n📝 SAMPLE DATA (5 tweets pertama):")
    print("-"*50)
    # Tampilkan sample dari df agar bisa membandingkan original vs cleaned
    for idx, row in df.head(5).iterrows():
        print(f"\nTweet #{idx+1}")
        # Mengambil 80 karakter pertama untuk preview
        orig_text = str(row['full_text'])[:80] if 'full_text' in row else "N/A"
        clean_txt = str(row['cleaned_text'])[:80]
        
        print(f"Original : {orig_text}...")
        print(f"Cleaned  : {clean_txt}...")
        print(f"Words    : {row['word_count']}")
    
    print(f"\n✅ Preprocessing selesai! File disimpan di: {output_file}")

# Main execution
if __name__ == "__main__":
    # File paths
    input_file = r"D:/Code bisa di D/MATKUL S5/Datmin/dataset_raw.csv"
    output_file = "dataset_raw_fulltext.csv"
    
    # Run preprocessing
    preprocess_data(input_file, output_file)
    
    print("\n🎉 Proses selesai! Data siap untuk analisis sentimen.")