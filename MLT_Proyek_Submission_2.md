# Laporan Proyek Machine Learning - Hasri Akbar Awal Rozaq

## Domain Proyek
Domain proyek yang dipilih dalam proyek *machine learning* ini adalah mengenai **Sistem Rekomendasi** dengan judul proyek "Penerapan Metode *Content-Based Filtering* dan *Collaborative Filtering* untuk Sistem Rekomendasi Buku Bacaan".

Buku merupakan balbal
Sebagai pembaca pasti akan bingung memilih buku mana lagi yang relevan dengan buku sebelumnya yang telah dibaca. Maka daripada itu, saya membuat sebuah sistem untuk merekomendasikan beberapa buku yang berkaitan dengan buku pembaca sebelumnya. Selain itu, saya juga membuat sistem rekomendasi kepada pengguna baru yang ingin membaca buku pertama kali.

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
  
- Untuk pembuatan sistem rekomendasi, saya memilih dua metode yaitu *content-based filtering* menggunakan algoritma *weighted-rating* dan *collaborative filtering* menggunakan algoritma K-NN (K-Nearest Neighbor). Pemilihan algoritma tersebut karena 

  Selain itu, berikut ini adalah kelebihan dan kelemahan algoritma KNN:
  
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
- Menghapus data yang kosong
    |     |     |
    | --- | --- |
    |Date |0    |
    |Close|29   |
    
    Menghapus data yang kosong adalah salah satu solusi untuk mengatasi *missing value*. Pada saat menganalisis data tersebut, ternyata nilai *null* merupakan data di hari libur dimana tidak ada perdagangan dalam hari tersebut. Maka, alangkah lebih baik untuk dihilangkan.
- Melakukan **pembagian** dataset menjadi dua bagian dengan persentase 80% untuk data latih dan 20% untuk data uji
    Pada proses pengujian model, maka perlu dilakukan pembagian dataset menjadi dua atau tiga bagian. Pada proyek ini dilakukan dua bagian saja yakni pada data latih dan data uji. Data latih terbagi dengan rasio 80% dari data asli, dimana dilakukan sepenuhnya untuk melatih model, sedangkan data uji terbagi dengan rasio 20% dari data asli merupakan data yang belum pernah dilihat oleh model dan diharapkan model dapat memiliki performa yang sama baiknya pada data uji seperti pada data latih. Karena pada dataset tersebut bersifat *univariate*, cara membagi data tersebut dengan membuat batasan data yang dijangkau.
- Melakukan **standarisasi data** pada fitur data
  Standarisasi dilakukan berfungsi untuk membuat komputasi dari pembuatan model dapat berjalan lebih cepat karena rentang datanya hanya antara 0-1. Ada berbagai cara standarisasi, akan tetapi pada pemodelan kali ini menggunakan MinMaxScaler. Berikut adalah rumus dari MinMaxScaler:

  <img src="https://user-images.githubusercontent.com/41296422/137363538-3d725636-fb74-4ec5-9f55-0fde810b5c71.png" width="30%" height="30%">

  Pada rumus tersebut, simbol `x` mewakili data yang diinputkan. MinMaxScaler sendiri bekerja dengan cara data asli akan dikurangi dengan data terkecil lalu dibagi dengan pengurungan dari data terbesar dan data terkecil.
- Penggunaan TimeseriesGenerator
  Data *time series* harus diubah menjadi struktur sampel dengan komponen *input* dan *output* sebelum dapat digunakan agar sesuai dengan *supervised learning model*. Ini bisa menjadi tantangan jika harus melakukan transformasi ini secara *manual*. TimeseriesGenerator salah satu solusi untuk mengubah data deret waktu *univariate* secara otomatis menjadi sampel, dan siap untuk melatih model *machine learning* [[8]](https://machinelearningmastery.com/how-to-use-the-timeseriesgenerator-for-time-series-forecasting-in-keras/).

## Modelling
Setelah melakukan pra-pemrosesan data yang baik pada tahap modeling akan dilakukan dua hal, yakni tahap pembuatan model *baseline* dan pembuatan model yang dikembangkan.
- Model *Baseline*
  Pada tahap ini saya membuat model dasar dengan menggunakan modul tensorflow yakni LSTM tanpa menggunakan parameter tambahan. Lalu melakukan prediksi kepada data ujinya.
- Model yang dikembangkan
  Kemudian setelah melihat kinerja model baseline, agar dapat bekerja lebih optimal lagi maka digunakan sebuah fungsi untuk mencari *hyperparameter* yang optimal dengan cara membuat *custom loss function*, menambahkan *layer* LSTM, dan penerapan *learning rate* pada *optimizer function*. Setelah ditemukan yang optimal, kemudian *hyperparameter* tersebut diterapkan ke model baseline.

Hasilnya dapat kita lihat pada grafik berikut ini:
**Model *Baseline***
![newplot (3)](https://user-images.githubusercontent.com/41296422/137366546-6eabe9a2-d759-4bcd-ad28-9e156d12c3ea.png)

**Model yang dikembangkan**
![newplot (2)](https://user-images.githubusercontent.com/41296422/137366585-2cff2287-6756-48c2-a670-fb9a1434631d.png)

Secara kasat mata, dari kedua model tersebut dapat memprediksi data uji dengan baik, akan tetapi kita harus memilih model manakah yang terbaik dengan cara mencari model dengan nilai *error rate* terkecil.

## Evaluating
Pada proyek ini, model yang dibuat merupakan kasus regresi dan menggunakan metriks perhitungan *Root Mean Squared Error* (RMSE) dan *Mean Absolute Error* (MAE). Penggunaan metriks tersebut karena memberikan bobot yang relatif tinggi untuk kesalahan besar. Berikut adalah rumus dari perhitungan RMSE dan MAE:

**RMSE**

<img src="https://user-images.githubusercontent.com/41296422/137367800-2dc7bd32-f39e-447e-915f-c623a5192a70.png" width="30%" height="30%">

Nilai RMSE didapatkan dari perhitungan jumlah setiap nilai prediksi dikurangi nilai asli dipangkat dua lalu dibagikan dengan banyaknya data dan terakhir diakarkan.

**MAE**

<img src="https://user-images.githubusercontent.com/41296422/137367898-3f9c131d-a300-4c70-a9bc-8e7d8d1458ee.gif" width="30%" height="30%">

Nilai MAE didapatkan dari perhitungan jumlah setiap nilai asli dikurangi nilai prediksi dipangkat dua lalu dibagikan dengan banyaknya data, dan nilai tersebut mutlak (tidak negatif).

Pada tabel di bawah ini adalah hasil dari perhitungan RMSE dan MAE dari kedua model di atas.

||Root Mean Squared Error|Mean Absolute Error|
|------|----------|-------|
|Model *Baseline*|0.002242|0.001706|
|Model yang dikembangkan|0.001476|0.001147|

Dapat disimpulkan bahwa, model yang dikembangkan lebih baik daripada model *baseline*.

## *References*

[[1]](https://ojs.unud.ac.id/index.php/bse/article/view/2195) Alwiyah dan Liyanto, “Analisis Teknikal Untuk Mendapatkan Profit,” Bul. Stud. Ekon., vol. 17, no. 2, hal. 221–228, 2012.

[[2]](https://doi.org/10.3390/a13080186)	M. S. Islam, E. Hossain, A. Rahman, M. S. Hossain, dan K. Andersson, “A Review on Recent Advancements in FOREX Currency Prediction,” Algorithms, vol. 13, no. 8, hal. 186, Jul 2020, https://doi.org/10.3390/a13080186.

[[3]](https://doi.org/10.24176/simet.v6i2.453)	R. H. Kusumodestoni dan S. Suyatno, “PREDIKSI FOREX MENGGUNAKAN MODEL NEURAL NETWORK,” Simetris  J. Tek. Mesin, Elektro dan Ilmu Komput., vol. 6, no. 2, hal. 205, Nov 2015, https://doi.org/10.24176/simet.v6i2.453.

[[4]](https://doi.org/10.1109/ICACCI.2017.8125846)	R. Vinayakumar, K. P. Soman, dan P. Poornachandran, “Long short-term memory based operation log anomaly detection,” in 2017 International Conference on Advances in Computing, Communications and Informatics (ICACCI), Sep 2017, hal. 236–242, https://doi.org/10.1109/ICACCI.2017.8125846.

[[5]](https://ntnuopen.ntnu.no/ntnu-xmlui/bitstream/handle/11250/2559922/18579_FULLTEXT.pdf?sequence=1)	S. Øyen, “Forecasting Multivariate Time Series Data Using Neural Networks,” Nor. Univ. Sci. Technol., no. June, 2018, [Daring]. Tersedia pada: https://ntnuopen.ntnu.no/ntnu-xmlui/bitstream/handle/11250/2559922/18579_FULLTEXT.pdf?sequence=1.

[[6]](https://doi.org/10.1007/978-981-10-2669-0)  H. Zhang, Q. Yan, G. Zhang, dan Z. Jiang, Theory, Methodology, Tools and Applications for Modeling and Simulation of Complex Systems, vol. 643, no. October. Singapore: Springer Singapore, 2016. https://doi.org/10.1007/978-981-10-2669-0.

[[7]](https://doi.org/10.1109/CVPR.2018.00572)	S. Li, W. Li, C. Cook, C. Zhu, dan Y. Gao, “Independently Recurrent Neural Network (IndRNN): Building A Longer and Deeper RNN,” Proc. IEEE Comput. Soc. Conf. Comput. Vis. Pattern Recognit., no. 1, hal. 5457–5466, 2018, https://doi.org/10.1109/CVPR.2018.00572.

[[8]](https://machinelearningmastery.com/how-to-use-the-timeseriesgenerator-for-time-series-forecasting-in-keras/)	J. Brownlee, “How to Use the TimeseriesGenerator for Time Series Forecasting in Keras,” 2018. https://machinelearningmastery.com/how-to-use-the-timeseriesgenerator-for-time-series-forecasting-in-keras/ (diakses Okt 13, 2021).
