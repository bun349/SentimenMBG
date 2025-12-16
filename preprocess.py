import pandas as pd
import re
import string
from langdetect import detect, LangDetectException
# Pastikan install dulu: pip install Sastrawi
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

# ==========================================
# 1. KONFIGURASI SASTRAWI (STOPWORD & STEMMER)
# ==========================================

# "Menghapus kata umum dengan Sastrawi Factory + daftar manual interjeksi"
factory_sw = StopWordRemoverFactory()
stopword_sastrawi = factory_sw.get_stop_words()

# Tambahan manual interjeksi/kata umum yang sering muncul di sosmed (sesuai PPT)
manual_stopwords = [
    'sih', 'dong', 'kok', 'deh', 'tuh', 'nih', 'ya', 'yah', 'loh', 
    'kan', 'kek', 'lah', 'pun', 'mah', 'si', 'itu', 'ini', 'yang', 
    'dan', 'di', 'ke', 'dari', 'untuk', 'pada'
]
# Gabungkan stopword (menggunakan set agar unik dan pencarian cepat)
final_stopwords = set(stopword_sastrawi + manual_stopwords)

# "Mengubah kata berimbuhan menjadi kata dasar"
factory_stem = StemmerFactory()
stemmer = factory_stem.create_stemmer()

# ==========================================
# 2. DEFINISI KAMUS SINGKATAN (MILIK ANDA)
# ==========================================
kamus_slang = {
    'yg': 'yang', 'y': 'ya', 'klo': 'kalau', 'tp': 'tapi', 'tpi': 'tapi',
    'tak': 'tidak', 'gak': 'tidak', 'ga': 'tidak', 'gk': 'tidak', 'nggak': 'tidak',
    'sdh': 'sudah', 'udh': 'sudah', 'dah': 'sudah',
    'dgn': 'dengan', 'krn': 'karena', 'karna': 'karena',
    'utk': 'untuk', 'unt': 'untuk',
    'bgt': 'banget', 'bngit': 'banget',
    'dlm': 'dalam', 'dr': 'dari',
    'jgn': 'jangan', 'tdk': 'tidak',
    'jd': 'jadi', 'jdi': 'jadi',
    'aja': 'saja', 'aj': 'saja',
    'krja': 'kerja', 'blm': 'belum',
    'bnyk': 'banyak', 'bnyak': 'banyak',
    'tmn': 'teman', 'org': 'orang',
    'mnrt': 'menurut', 'dpt': 'dapat',
    'pke': 'pakai', 'pake': 'pakai',
    'sm': 'sama', 'smua': 'semua',
    'nnti': 'nanti', 'ntar': 'nanti',
    'bs': 'bisa', 'bsa': 'bisa',
    'ak': 'aku', 'aq': 'aku', 'gw': 'aku', 'gue': 'aku', 'sy': 'saya',
    'km': 'kamu', 'lu': 'kamu', 'loe': 'kamu',
    'knp': 'kenapa', 'np': 'kenapa',
    'gini': 'begini', 'gitu': 'begitu',
    'kpn': 'kapan', 'dmn': 'dimana',
    'jg': 'juga', 'jga': 'juga',
    'mkn': 'makan', 'mkan': 'makan',
    'lbh': 'lebih', 'kurleb': 'kurang lebih',
    'skrg': 'sekarang', 'trus': 'terus',
    'bgmn': 'bagaimana', 'gmn': 'bagaimana',
    'dtg': 'datang', 'tuh': 'itu', 'ni': 'ini',
    'ok': 'oke', 'oke': 'oke', 'sip': 'siap',
    'thn': 'tahun', 'bln': 'bulan', 'mgu': 'minggu',
    'bgs': 'bagus', 'jlk': 'jelek', 'parah': 'buruk',
}

# ==========================================
# 3. FUNGSI PREPROCESSING (PIPELINE)
# ==========================================

def clean_regex(text):
    """Tahap 1: Cleaning (PPT Source 87)"""
    if not isinstance(text, str): return ""
    text = text.lower()
    text = re.sub(r'http\S+|www\S+|https\S+', '', text) # Hapus URL
    text = re.sub(r'@\w+', '', text) # Hapus Mention
    text = re.sub(r'#\w+', '', text) # Hapus Hashtag
    text = re.sub(r'\d+', '', text)  # Hapus Angka
    text = text.translate(str.maketrans('', '', string.punctuation)) # Hapus Tanda Baca
    text = re.sub(r'\s+', ' ', text).strip() # Hapus spasi dobel
    return text

def normalize_slang(text):
    """Tahap 2: Normalisasi Slang (PPT Source 88)"""
    if not text: return ""
    words = text.split()
    normalized_words = [kamus_slang.get(word, word) for word in words]
    return " ".join(normalized_words)

def remove_stopwords(text):
    """Tahap 3: Stopword Removal (PPT Source 89)"""
    if not text: return ""
    words = text.split()
    # Hapus kata jika ada di set stopword
    filtered_words = [word for word in words if word not in final_stopwords]
    return " ".join(filtered_words)

def stemming_text(text):
    """Tahap 4: Stemming (PPT Source 90)"""
    # Mengubah kata berimbuhan jadi kata dasar
    # Warning: Proses ini biasanya memakan waktu paling lama
    if not text: return ""
    return stemmer.stem(text)

# ==========================================
# 4. FUNGSI FILTERING (VALIDASI)
# ==========================================
# Fungsi ini tetap ada tapi tidak akan dipanggil di main()

def is_gibberish(text):
    """Cek kata asal-asalan"""
    if not re.search(r'[aiueoAIUEO]', text): return True
    if re.search(r'(.)\1{4,}', text): return True
    return False

def is_valid_content(text):
    """Tahap 5: Validasi Akhir"""
    if not isinstance(text, str) or not text.strip(): return False
    
    words = text.split()
    if len(words) < 3: return False 
    
    if is_gibberish(text): return False
    
    # Cek Bahasa
    try:
        if detect(text) != 'id': return False
    except LangDetectException:
        return False
        
    return True

# ==========================================
# 5. EKSEKUSI UTAMA
# ==========================================

def main():
    output_file = 'data_final_preprocessing_no_filter.csv'
    
    print(f"Membaca data...")
    try:
        df = pd.read_csv("dataset_sentimen - raw.csv")
        # Deteksi nama kolom teks
        col_text = next((col for col in df.columns if any(x in col.lower() for x in ['text', 'content', 'caption', 'cleaned'])), df.columns[0])
        print(f"Menggunakan kolom teks: {col_text}")
        
        df[col_text] = df[col_text].astype(str)
        print(f"Total data awal: {len(df)}")
        
        # --- PIPELINE SESUAI PPT (HAL 12) ---
        
        # 1. Cleaning
        print("1. Melakukan Cleaning Regex (Hapus URL, Emoji, dll)...")
        df['step1_clean'] = df[col_text].apply(clean_regex)
        
        # 2. Normalisasi Slang
        print("2. Melakukan Normalisasi Slang...")
        df['step2_normal'] = df['step1_clean'].apply(normalize_slang)
        
        # 3. Stopword Removal (BARU)
        print("3. Melakukan Stopword Removal (Sastrawi + Manual)...")
        df['step3_stopword'] = df['step2_normal'].apply(remove_stopwords)
        
        # 4. Stemming (BARU)
        print("4. Melakukan Stemming (Sastrawi) - Proses ini mungkin agak lama...")
        df['step4_stemmed'] = df['step3_stopword'].apply(stemming_text)
        
        # mask_valid = df['step4_stemmed'].apply(is_valid_content)
        # df_final = df[mask_valid].copy()
        
        df_final = df.copy()
        
        # --- SIMPAN HASIL ---
        # Simpan kolom hasil akhir sebagai 'processed_text'
        df_final['processed_text'] = df_final['step4_stemmed']
        
        # Pilih kolom yang mau disimpan
        cols_to_save = [col_text, 'processed_text']
        if 'label' in df_final.columns: cols_to_save.append('label')
            
        print(f"Total data disimpan: {len(df_final)}")
        
        df_final[cols_to_save].to_csv(output_file, index=False)
        
        print("\n" + "="*40)
        print("PROSES SELESAI!")
        print(f"File disimpan: {output_file}")
        print("="*40)
        
        # Tampilkan contoh perbandingan
        print("\nContoh Perubahan:")
        print(df_final[[col_text, 'processed_text']].head(5))

    except FileNotFoundError:
        print("File tidak ditemukan. Pastikan path file benar.")
    except Exception as e:
        print(f"Terjadi error: {e}")

if __name__ == "__main__":
    main()