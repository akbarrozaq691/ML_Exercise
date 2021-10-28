# Laporan Proyek Machine Learning - Hasri Akbar Awal Rozaq

## Domain Proyek
Domain proyek yang dipilih dalam proyek *machine learning* ini adalah mengenai **Sistem Rekomendasi** dengan judul proyek "Penerapan Metode *Content-Based Filtering* dan *Collaborative Filtering* untuk Sistem Rekomendasi Buku Bacaan".

Buku merupakan jendela ilmu dimana sangat penting bagi kehidupan manusia [[1]](https://journal.unhas.ac.id/index.php/jupiter/article/view/1672). Dengan terbiasa membaca maka 
seseorang akan memiliki cakrawala pengetahuan yang luas, kreativitas terbuka, imajinasi tinggi, pemikiran yang maju dan berkembang serta menjadi cikal bakal pemberdayaan manusia yang cerdas dan berintelektual. Tingkat kesadaran membaca buku terutama di Indonesia itu sangat rendah [[2]](http://jurnal.unpad.ac.id/jkip/article/download/10003/4723). Rata-rata nasional distribusi literasi pada kemampuan membaca pelajar di Indonesia adalah 46,83% berada pada kategori Kurang, hanya 6,06% berada pada kategori Baik, dan 47,11 berada pada kategori Cukup [[3]](https://core.ac.uk/download/pdf/287170379.pdf). Padahal, di negara lain membaca buku sudah menjadi budaya sehari-hari dimana kegiatan tersebut dimanfaatkan ketika waktu mereka kosong. Dengan adanya teknologi saat ini, diharapkan mampu meningkatkan budaya membaca dari setiap kalangan. Penggunaan media digital sudah sangat melekat setiap harinya. Hasil penelitian yang telah dilakukan mengungkap fakta bahwa ada perbedaan signifikan setelah melakukan kegiatan membaca dengan media digital [[4]](http://eprints.ukmc.ac.id/3028/4/Artikel%20%28Ira%20Irzawati%29.pdf).

Penggunaan media baca selalu berperan penting di dalamnya. Pengguna aktif akan selalu membaca buku digital secara berkala. Akan tetapi, sebagai pembaca pasti akan bingung memilih buku mana lagi yang relevan dengan buku sebelumnya yang telah dibaca. Dari permasalahan tersebut, saya akan membuat sebuah sistem untuk merekomendasikan beberapa buku yang berkaitan dengan buku pembaca sebelumnya dengan metode *collaborative filtering*. Selain itu, saya juga membuat sistem rekomendasi kepada pengguna baru yang ingin membaca buku pertama kali dengan metode *content-based filtering* yang nantinya akan diterapkan pada salah satu media digital.

## Business Understanding
### Problem Statements
Berdasarkan kondisi yang telah diuraikan sebelumnya, berikut ini merupakan rincian masalah yang dapat diselesaikan:
- Bagaimana cara membuat sistem rekomendasi buku dengan metode *content-based filtering* dan *collaborative filtering*?
- Apakah kedua metode tersebut efektif untuk digunakan?

### Goals
Untuk menjawab pertanyaan pada *problem statements* di atas, berikut tujuan dari dibuatnya proyek ini:
- Model yang dipilih dapat memberikan rekomendasi buku yang akan dibaca
- Dapat menyimpulkan keefektifan metode yang digunakan dengan menghitung nilai *error rate*

### Solutions Approach
Solusi yang dilakukan untuk memenuhi tujuan dari proyek ini di antaranya:
- Untuk pra-pemrosesan data dapat dilakukan beberapa teknik, antara lain:
  - Pembersihan data yang unik
  - Pembersihan data yang *missing value*
  - Pembersihan data yang memiliki nilai *rating* adalah 0
  
  Poin pra-pemrosesan data akan dijelaskan secara rinci pada bagian `Data Preparation`.
  
- Untuk pembuatan sistem rekomendasi, saya memilih dua metode yaitu *content-based filtering* menggunakan algoritma *weighted-rating* dan *collaborative filtering* menggunakan algoritma K-NN (*K-Nearest Neighbor*). Pemilihan kedua algoritma tersebut karena dapat mengatasi masalah klasifikasi terutama untuk sistem rekomendasi buku [[5]](https://media.neliti.com/media/publications/127887-ID-implementasi-metode-k-nearest-neighbor-k.pdf).

  - *Weighted-Rating*
    
    *Weighted-Rating* atau biasa disebut dengan *Weighted Scoring System* merupakan sistem penilaian yang didasarkan pada kriteria-kriteria penilaian, *weight*, *rating*, serta *score*-nya. Sistem ini biasanya digunakan oleh para pembuat keputusan yang dihadapkan pada sejumlah kriteria yang telah ditetapkan sebagai bahan pertimbangan dalam pengambilan keputusan [[6]](https://media.neliti.com/media/publications/79738-ID-implementasi-weight-scoring-system-dalam.pdf). Sehingga, cocok digunakan untuk kasus *content-based filtering* ini. Rumus dari algoritma tersebut dapat diartikan sebagai berikut sebagai berikut [[7]](https://jurnal.iaii.or.id/index.php/RESTI/article/view/834):
    
    <img src="https://user-images.githubusercontent.com/41296422/139244529-2aec8fe4-e4ba-4655-8980-cd15e54535b2.JPG" width="50%" height="50%">
    
    Dimana dapat diartikan:
    - `v` adalah jumlah voting untuk sebuah buku
    - `m` adalah suara minimum yang diperlukan untuk dicantumkan dalam bagan
    - `R` adalah nilai rata-rata dari rating sebuah buku
    - `C` adalah suara rata-rata di seluruh laporan pada sebuah data
    
    Dengan algoritma ini, dapat memberikan hasil rekomendasi buku dengan cara mengurutkan mulai dari bobot tertinggi hingga terendah. Selain itu, berikut ini adalah kelebihan dan kelemahan dari algoritma ini [[8]](https://jurnal.kwikkiangie.ac.id/index.php/JIB/article/view/711/463):
    - Kelebihan
      - Konsep yang sederhana untuk melakukan pembobotan
      - Mempercepat proses perhitungan kriteria
      - Dapat digunakan untuk pengambilan *single* dan keputusan *multidimensional*
    - Kelemahan
      - Metode ini hanya metode matematis tanpa ada pengujian secara statistik
    
  - K-NN (*K-Nearest Neighbor*)
  
    K-NN singkatan dari *K-Nearest Neighbor* dimana merupakan sebuah metode untuk melakukan klasifikasi terhadap objek berdasarkan data pembelajaran yang jaraknya paling dekat dengan objek tersebut [[9]](https://journal.unhas.ac.id/index.php/jmsk/article/download/3399/1936). Sehingga sangat cocok untuk metode *collaborative filtering* saat ini. Berikut adalah cara kerja algoritma ini (diterjemahkan dari [[10]](https://www.researchgate.net/publication/338718633_Optimization_of_K_Value_at_the_K-NN_algorithm_in_clustering_using_the_expectation_maximization_algorithm)):
    - Memasukkan data
    - Inisialisasi nilai K (banyaknya tetangga / kelompok)
    - Kalkukasi jarak dengan euclidian dengan rumus sebagai berikut:
      
      <img src="https://user-images.githubusercontent.com/41296422/139250571-4744d4b3-39c8-40d9-b239-2f5d8fceda5e.png" width="25%" height="25%">
      
    - Mengurutkan hasil kalkulasi jarak
    - Memilih alternatif yang paling banyak
    - Hasil penentuan data berdasarkan nilai yang telah dihitung sebelumnya
  
    Dengan algoritma ini, dapat memberikan hasil user terdekat dengan user yang dipilih dan rekomendasi buku dengan cara mencari buku yang sesuai dengan user yang lain juga. Selain itu, berikut ini adalah kelebihan dan kelemahan algoritma K-NN [[11]](https://www.researchgate.net/publication/291457761_Optimasi_Teknik_Klasifikasi_Modified_k_Nearest_Neighbor_Menggunakan_Algoritma_Genetika):
    - Kelebihan:
      - Sederhana dan mudah dipelajari
      - Pelatihan sangat cepat
      - tahan terhadap data yang memiliki derau
    - Kekurangan:
      - Komputasi Kompleks
      - Keterbatasan Memori
  
## Data Understanding
![book](https://user-images.githubusercontent.com/41296422/139196256-e84cc6f6-8f6f-4c98-8dfe-011a8619540a.JPG)

Informasi Dataset:

Dataset: [Book Recommendation Dataset](https://www.kaggle.com/arashnic/book-recommendation-dataset)

|Jenis                    |Keterangan                                                                                                |
| ----------------------- |  -----------------------------------------------------------------------------------------------------   |
|Sumber                   |[Kaggle Dataset: Book Recommendation Dataset](https://www.kaggle.com/arashnic/book-recommendation-dataset)|
|Lisensi                  |CC0: Public Domain                                                                                        |
|Kategori                 |Literature, Culture, and Humanities                                                                       |
|Jenis dan Ukuran Berkas  |Zip (107MB)                                                                                               |

Terdapat 3 file yang didownload yaitu: `Books.csv`, `Ratings.csv`, `Users.csv`. Berikut adalah rincian dari ketiga file tersebut:

1. `Books.csv`
    
    Pada file dataset tersebut, berisi informasi metriks data buku dengan jumlah 271.360 data. Terdapat 7 buah data bertipe *object*. Dataset tersebut memiliki data kosong pada kolom ***Year of Publication*** dan ***Image-URL-L***. Untuk mengenal variabel apa saja pada dataset tersebut, dapat dilihat rincian sebagai berikut:
    - `ISBN`: Kode pengidentifikasian buku yang bersifat unik (kode buku)
    - `Book-Title`: Judul dari buku yang ada
    - `Book-Author`: Penulis dari buku
    - `Year-Of-Publication`: Tahun diterbitnya buku
    - `Publisher`: Tempat dimana buku tersebut dicetak
    - `Image-URL-S`: Alamat URL gambar sampul buku ukuran kecil
    - `Image-URL-M`: Alamat URL gambar sampul buku ukuran sedang
    - `Image-URL-L`: Alamat URL gambar sampul buku ukuran besar

2. `Ratings.csv`

    Pada file dataset tersebut, berisi informasi metriks data rating buku dengan jumlah 1.149.780 data. Terdapat 2 buah data bertipe *integer* dan 1 data bertipe *object*. Dataset tersebut tidak memiliki data kosong. Untuk mengenal variabel apa saja pada dataset tersebut, dapat dilihat rincian sebagai berikut:
   - `ISBN`: Kode pengidentifikasian buku yang bersifat unik (kode buku), nantinya akan direlasikan dengan dataset `books.csv`
   - `User-ID`: ID dari pengguna sebelumnya
   - `Book-Rating`: Penilaian tentang buku tersebut

3. `Users.csv`

    Pada file dataset tersebut, berisi informasi metriks data pengguna buku dengan jumlah 278.858 data. Terdapat 1 buah data bertipe *integer*, 1 buah data bertipe *float*, dan 1 data bertipe *object*. Dataset tersebut memiliki data kosong pada kolom ***Age*** / umur. Untuk mengenal variabel apa saja pada dataset tersebut, dapat dilihat rincian sebagai berikut:
   - `User-ID`: ID dari pengguna sebelumnya, nantinya akan direlasikan dengan dataset `ratings.csv`
   - `Location`: Tempat asal dari pengguna
   - `Age`: Umur dari pengguna

Pada kasus kali ini, sebelum melakukan tahap pengujian data, 3 data tersebut akan disatukan terlebih dahulu. Setelah penggabungan data tersebut, saya memilih variabel `User-ID`, `ISBN`, `Book-Rating`, `Book-Title`, `Book-Author`, dan `Year-Of-Publication` karena pada saat pemodelan nantinya akan diberikan rekomendasi dengan acuan rating buku.

Kemudian terdapat juga visualisasi data untuk kolom `Book-Rating`:
![book-rating](https://user-images.githubusercontent.com/41296422/139227431-d468d984-47db-4d6b-b1e3-22987ce8910b.png)

## Data Preparation
Berikut adalah tahapan pra-pemrosesan data seperti yang telah dijelaskan pada *solution statements*:

- Pembersihan data yang unik
  
  Pembersihan data yang unik dilakukan karena adanya kejanggalan pada data di kolom `Year-of-Publication`. Data tersebut dapat dilihat sebagai berikut:
  
  ![Data Unik](https://user-images.githubusercontent.com/41296422/139228416-1d67cd4a-21fb-4ee1-8ac7-201269c56f91.JPG)
  
  Dapat kita lihat bahwa pada kolom tersebut ada ketidaksesuaian data yaitu **DK Publishing INC** dan **Gallimard**. Maka, data yang berindeks dengan nilai tersebut akan dihapus.
    
- Pembersihan data yang kosong (*missing value*)

  Setelah penyatuan data pada proses sebelumnya dilakukan, ternyata masih ada data yang kosong pada kolom `Book-Author`
  
  ![data kosong](https://user-images.githubusercontent.com/41296422/139229230-e930ecb7-311f-494e-93a1-f31629f66475.JPG)
    
    Karena pada data tersebut hanya ada satu data saja yang kosong, maka saya menghapusnya. Proses penghapusan data yang kosong adalah salah satu solusi untuk mengatasi *missing value*.
    
- Pembersihan data yang memiliki rating 0

  Proses ini dilakukan berfungsi untuk mengoptimalkan sistem rekomendasi buku dimana dapat dikatakan bahwa buku yang belum diberi rating maka bisa jadi belum sesuai dengan standar rekomendasi. Rating yang saya gunakan untuk sistem ini adalah dari 1 - 10.

## Modelling
Setelah melakukan pra-pemrosesan data, selanjutnya adalah membuat sistem rekomendasi buku dengan metode *content-based filtering* menggunakan algoritma *weighted-rating* dan *collaborative filtering* menggunakan algoritma K-NN (*K-Nearest Neighbor*).
- Menggunakan *Weighted-Rating*

  Pada proses rekomendasi buku yaitu *content-based filtering*, saya membuat model dengan menggunakan algoritma *weighted-rating* dimana model tersebut akan mencari buku terbaik dengan cara mengurutkan bobot nilai rating dari sebuah buku mulai dari terbesar hingga terkecil. Pada gambar di bawah ini, saya mencontohkan 10 buku terbaik untuk bisa dibaca oleh pengguna.

  ![data bobot](https://user-images.githubusercontent.com/41296422/139231834-f3c637bd-4bf0-4936-882b-f99f2f581695.JPG)
  
- Menggunakan K-NN (*K-Nearest Neighbor*)

  Selanjutnya, pada proses rekomendasi buku yaitu *collaborative filtering*, saya membuat model dengan menggunakan algoritma K-NN dimana model tersebut akan mencari buku yang relevan dengan buku yang sudah dibaca oleh pengguna. Algoritma ini bekerja mengacu dari nilai rating buku yang sudah dibaca pengguna dengan 1000 buku terbaik yang lainnya dan kesamaan buku yang telah dibaca oleh pengguna lain. Berikut adalah contoh rekomendasi buku untuk salah satu pengguna dengan memperhatikan kesamaan buku yang telah dibaca oleh 5 pengguna lain.
  
  ![pengguna mirip](https://user-images.githubusercontent.com/41296422/139233346-ab78d4b9-bc5f-4a49-9909-81d027649917.JPG)
  
  ![buku rekomendasi](https://user-images.githubusercontent.com/41296422/139233997-cc541d2d-f843-41f3-8edc-41180ac3e46f.JPG)

## Evaluating
Pada proyek ini, perhitungan evaluasi untuk algoritma K-NN menggunakan metriks *Root Mean Squared Error* (RMSE) dan *Mean Absolute Error* (MAE). Penggunaan metriks tersebut karena memberikan bobot yang relatif tinggi untuk kesalahan besar. Berikut adalah rumus dari perhitungan RMSE dan MAE:

**RMSE**

<img src="https://user-images.githubusercontent.com/41296422/137367800-2dc7bd32-f39e-447e-915f-c623a5192a70.png" width="25%" height="25%">

Nilai RMSE didapatkan dari perhitungan jumlah setiap nilai prediksi dikurangi nilai asli dipangkat dua lalu dibagikan dengan banyaknya data dan terakhir diakarkan.

**MAE**

<img src="https://user-images.githubusercontent.com/41296422/137367898-3f9c131d-a300-4c70-a9bc-8e7d8d1458ee.gif" width="25%" height="25%">

Nilai MAE didapatkan dari perhitungan jumlah setiap nilai asli dikurangi nilai prediksi dipangkat dua lalu dibagikan dengan banyaknya data, dan nilai tersebut mutlak (tidak negatif).

Pada tabel di bawah ini adalah hasil dari perhitungan RMSE dan MAE dari nilai rating buku yang telah dibaca pengguna dengan nilai rating buku rekomendasi.

||Root Mean Squared Error|Mean Absolute Error|
|------|----------|-------|
|Model K-NN|2.1909|2.0000|

Dapat disimpulkan bahwa, algoritma *weighted-rating* dan K-NN dapat digunakan untuk memberikan rekomendasi buku. Tetapi, melihat nilai *error rate* yang ada pada tabel di atas masih sangat memungkinkan untuk menguranginya, alangkah baiknya dapat dikembangkan dengan metode *hyperparameter tuning* atau menggunakan algoritma lain misalnya DNN (*Deep Neural Network*).

## Referensi

