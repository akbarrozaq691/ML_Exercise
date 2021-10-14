# Laporan Proyek Machine Learning - Hasri Akbar Awal Rozaq

## Domain Proyek
Domain proyek yang dipilih dalam proyek *machine learning* ini adalah mengenai **keuangan** dengan judul proyek "Prediksi Nilai Tukar Uang EUR terhadap USD".

*Foreign exchange* (Forex) adalah pasar yang mengkhususkan diri dalam perdagangan pertukaran valuta asing. Dalam pasar *forex* sendiri seseorang yang melakukan perdagangan biasa disebut dengan *trader* memiliki dua pilihan, baik membeli atau menjual mata uang yang diperdagangkan. Jika kurs jual mata uang lebih besar dari tingkat pembelian, itu menghasilkan keuntungan bagi orang tersebut [(Islam et.al., 2020)](https://www.researchgate.net/publication/343342034). Nilai tukar mata uang yang sering tidak stabil menjadikan bisnis *forex* sebagai bisnis beresiko tinggi tetapi juga sebagai bisnis dengan keuntungan besar pula sehingga nilai pertukaran yang dialami perlu diperhatikan. Dari aspek tersebut maka diperlukan proses prediksi yang tepat untuk meminimalkan resiko dan meningkatkan sebuah keuntungan [(Kusumodestoni, 2015)](https://jurnal.umk.ac.id/index.php/simet/article/view/453). Data tersebut bersifat *time-series* (sekuensial) sehingga perlu algoritma yang dapat mengatasi masalah tersebut. Sehingga, teknik *predictive modelling* sangat cocok digunakan untuk diterapkan pada kasus tersebut.

## Business Understanding
### Problem Statements
Berdasarkan kondisi yang telah diuraikan sebelumnya, perancang algoritma ingin membantu *trader* dalam mengembangkan sistem prediksi *forex* untuk menjawab permasalahan tersebut.
- Bagaimana cara kerja algoritma *time-series* memprediksi data forex?
- Algoritma manakah yang terbaik untuk mengatasi masalah tersebut?

### Goals
Untuk menjawab pertanyaan tersebut, saya akan membuat predictive modelling dengan *goals* sebagai berikut:
- Algoritma yang dipilih dapat memprediksi nilai tukar mata uang dengan rentang waktu tertentu.
- Dapat menemukan algoritma terbaik terhadap data forex dengan membandingkan nilai error rate terkecil dari model yang ada.

### Solutions Statements
Solusi yang dilakukan untuk memenuhi tujuan dari proyek ini di antaranya:
- Untuk pra-pemrosesan data dapat dilakukan beberapa teknik, antara lain:
  - Menghapus data yang kosong
  - Melakukan **pembagian** dataset menjadi dua bagian dengan persentase 80% untuk data latih dan 20% untuk data uji
  - Melakukan **standarisasi data** pada fitur data
  - Karena data bersifat *time-series*, maka alangkah lebih baik diubah menjadi data sekuensial menggunakan TimeseriesGenerator
  
  Poin pra-pemrosesan data akan dijelaskan secara rinci pada bagian `Data Preparation`.
- Untuk pembuatan model sendiri menggunakan algoritma **LSTM (Long-Short Term Memory)** sebagai model _baseline_. Algoritma tersebut dipilih karena mudah diimplementasikan dan juga cocok untuk kasus data sekuensial (_time series_). Algoritma ini dapat mengingat masa lalu untuk memprediksi masa depan melalui _nodes_ pengingat. Cara kerja algoritma ini adalah sebagai berikut (diterjemahkan dari [(Luwiji, 2020)](https://pypi.org/project/luwiji/)):
  - Data akan masuk ke dalam _input gate_ terlebih dahulu
  - Data yang telah masuk akan dipelajari pada _short term memory_ 
  - Pada ingatan lama (_long term memory_), data yang tidak berguna akan dilupakan
  - Untuk mengingat kembali, ingatan lama yang belum dilupakan ditambahkan dengan data yang sedang dipelajari akan menjadi _long term memory_ yang baru
  - Untuk memprediksi, ingatan lama yang belum dilupakan ditambahkan dengan data yang sedang dipelajari dan akan digunakan, lalu menjadi _short term memory_ yang baru
  - Data yang diprediksi akan dikeluarkan pada _output gate_

  Selain itu, berikut ini adalah kelebihan dan kelemahan algoritma LSTM:
  - Kelebihan (diterjemahkan dari [(Zhang, 2016)](https://www.springer.com/gp/book/9789811026652)):
    - Adanya arsitektur mengingat dan melupakan output yang akan diproses kembali menjadi input
    - Dapat mempertahankan error yang terjadi ketika melakukan backpropagation sehingga tidak memungkinkan kesalahan meningkat
  - Kekurangan (diterjemahkan dari [(Li, 2018)](https://arxiv.org/abs/1803.04831)):
    - Memiliki arsitektur yang kompleks sehingga beban komputasi menjadi tinggi terutama ketika diterapkan pada kasus skala besar
- Kemudian model _baseline_ tersebut dikembangkan dengan pengaturan _hyperparameter_ dengan cara membuat _custom loss function_, menambahkan _layer_ LSTM, dan penerapan _learning rate_ pada _optimizer function_ dimana dengan memodifikasi parameter tersebut, dapat mengurangi nilai _error_. Penggunaan _custom loss function_ diambil dari metode yang bernama _Huber_. Berikut adalah rumus dari _Huber Loss Function_

    ![Capture](https://user-images.githubusercontent.com/41296422/137327705-7799a336-9a43-4d24-9c9e-660b137d8fa0.JPG)

    Cara kerja dari metode tersebut antara lain:
     - Menghitung nilai _error_ terlebih dahulu dimana nilai tersebut didapatkan dari pengurangan antara data asli dan data prediksi
     - Menghitung nilai _loss_ terkecil dengan cara nilai _error_ pangkat 2 dibagi 2
     - Menghitung nilai _loss_ terbesar dengan cara nilai _thershold_ dikali dengan nilai _error_ mutlak dikurangi 1/2 dari nilai _threshold_
     - Lalu akan dicari nilai _loss_ sesuai _thershold_
  Dengan adanya pengaturan _hyperparameter_ ini harapannya akan menciptakan model yang lebih akurat dan memiliki nilai _error rate_ kecil. 
## Data Understanding
![Capture](https://user-images.githubusercontent.com/41296422/137277185-d5e6a42d-47e9-4468-bacf-90e2a0c2399e.JPG)

Informasi Dataset:

Dataset: [EURUSD=X](https://finance.yahoo.com/quote/EURUSD%3DX/history?period1=1070236800&period2=1634083200&interval=1d&filter=history&frequency=1d&includeAdjustedClose=true)

|Jenis                    |Keterangan                                                                                              |
| ----------------------- |  ----------------------------------------------------------------------------------------------------- |
|Sumber                   |[yahoo! Finance: EUR/USD (EURUSD=X)](https://finance.yahoo.com/quote/EURUSD=X?p=EURUSD=X&.tsrc=fin-srch)|
|Lisensi                  |CC0: Public Domain                                                                                      |
|Kategori                 |Finance (Keuangan)                                                                                      |
|Jenis dan Ukuran Berkas  |CSV (263K)                                                                                              |
|Time Frame yang Digunakan|Daily                                                                                                   |
|Range Waktu              |1 Desember 2003 - 13 Oktober 2021                                                                       |

Pada berkas yang diunduh yakni `EURUSD=X.csv` berisi informasi metriks nilai tukar uang EUR/USD dengan jumlah 4663 data. Terdapat 6 buah data numerik (tipe data float64) dan 1 buah data _date time_ (tipe data datetime64). Dataset tersebut memiliki data kosong kecuali pada kolom tanggal. Untuk mengenal variabel apa saja pada dataset tersebut, dapat dilihat pada poin-poin sebagai berikut:
1. `Date`: Tanggal dimana menunjukkan waktu terjadinya pembukaan dan penutup harga, pada dataset kali ini, data tersebut berisi waktu harian dan sangat penting untuk dianalisis apakah harga naik / turun dalam satu hari terakhir
2. `Open`: Harga pertama kali transaksi dilakukan pada hari itu. Harga _open_ tersebut mencerminkan semua informasi pasar yang ada, yang terjadi atau muncul diantara harga penutupan sehari sebelumnya dan ketika saat-saat terakhir pemodal boleh memasukkan order ke mesin bursa.
3. `High`: Kisaran harga pergerakan harian dari saham tersebut dimana pemodal memiliki keberanian atau rasionalitas untuk melakukan posisi beli.
4. `Low`: Kisaran harga pergerakan harian dari saham tersebut dimana pemodal memiliki keberanian atau rasionalitas untuk melakukan posisi jual.
5. `Close`: Harga close ini mencerminkan semua informasi yang ada pada semua pelaku pasar (terutama pelaku pasar institusi yang memiliki informasi yang lebih akurat) pada saat perdagangan saham tersebut berakhir.
6. `Adjusted Close`: Seperti halnya variabel close, variabel adjust close merupakan harga penutupan yang disesuaikan.
7. `Volume`: Representasi dari aktivitas yang berlangsung selama suatu periode trading.

Pada kasus kali ini yang akan diterapkan adalah variabel `Date` dan `Close` karena keduanya bisa memicu signal beli atau signal jual untuk waktu yang akan datang.
Kemudian terdapat juga visualisasi data untuk kolom `Close` dengan `Date` sebagai indeksnya:
![newplot (1)](https://user-images.githubusercontent.com/41296422/137333358-6fb353be-5227-4e1e-9895-881e86f51f3b.png)

## Data Preparation
Berikut adalah tahapan pra-pemrosesan data seperti yang telah dijelaskan pada _solution statements_:
- Menghapus data yang kosong
    |     |     |
    | --- | --- |
    |Date |0    |
    |Close|29   |
    
    Menghapus data yang kosong adalah salah satu solusi untuk mengatasi _missing value_. Pada saat menganalisis data tersebut, ternyata nilai _null_ merupakan data di hari libur dimana tidak ada perdagangan dalam hari tersebut. Maka, alangkah lebih baik untuk dihilangkan.
- Melakukan **pembagian** dataset menjadi dua bagian dengan persentase 80% untuk data latih dan 20% untuk data uji
    Pada proses pengujian model, maka perlu dilakukan pembagian dataset menjadi dua atau tiga bagian. Pada proyek ini dilakukan dua bagian saja yakni pada data latih dan data uji. Data latih terbagi dengan rasio 80% dari data asli, dimana dilakukan sepenuhnya untuk melatih model, sedangkan data uji terbagi dengan rasio 20% dari data asli merupakan data yang belum pernah dilihat oleh model dan diharapkan model dapat memiliki performa yang sama baiknya pada data uji seperti pada data latih. Karena pada dataset tersebut bersifat _univariate_, cara membagi data tersebut dengan membuat batasan data yang dijangkau.
- Melakukan **standarisasi data** pada fitur data
  Standarisasi dilakukan berfungsi untuk membuat komputasi dari pembuatan model dapat berjalan lebih cepat karena rentang datanya hanya antara 0-1. Ada berbagai cara standarisasi, akan tetapi pada pemodelan kali ini menggunakan MinMaxScaler. Berikut adalah rumus dari MinMaxScaler:
  ![EuitP](https://user-images.githubusercontent.com/41296422/137363538-3d725636-fb74-4ec5-9f55-0fde810b5c71.png)

  Pada rumus tersebut, simbol `x` mewakili data yang diinputkan. MinMaxScaler sendiri bekerja dengan cara data asli akan dikurangi dengan data terkecil lalu dibagi dengan pengurungan dari data terbesar dan data terkecil.
- Penggunaan TimeseriesGenerator
  Data _time series_ harus diubah menjadi struktur sampel dengan komponen _input_ dan _output_ sebelum dapat digunakan agar sesuai dengan _supervised learning model_. Ini bisa menjadi tantangan jika harus melakukan transformasi ini secara _manual_. TimeseriesGenerator salah satu solusi untuk mengubah data deret waktu _univariate_ secara otomatis menjadi sampel, dan siap untuk melatih model _deep learning_ [(Machine learning mastery, 2020)](https://machinelearningmastery.com/how-to-use-the-timeseriesgenerator-for-time-series-forecasting-in-keras/).

## Modelling
Setelah melakukan pra-pemrosesan data yang baik pada tahap modeling akan dilakukan dua hal, yakni tahap pembuatan model _baseline_ dan pembuatan model yang dikembangkan.
- Model _Baseline_
  Pada tahap ini saya membuat model dasar dengan menggunakan modul tensorflow yakni LSTM tanpa menggunakan parameter tambahan. Lalu melakukan prediksi kepada data ujinya.
- Model yang dikembangkan
  Kemudian setelah melihat kinerja model baseline, agar dapat bekerja lebih optimal lagi maka digunakan sebuah fungsi untuk mencari _hyperparameter_ yang optimal dengan cara membuat _custom loss function_, menambahkan _layer_ LSTM, dan penerapan _learning rate_ pada _optimizer function_. Setelah ditemukan yang optimal, kemudian _hyperparameter_ tersebut diterapkan ke model baseline.

Hasilnya dapat kita lihat pada grafik berikut ini:
**Model _Baseline_**
![newplot (3)](https://user-images.githubusercontent.com/41296422/137366546-6eabe9a2-d759-4bcd-ad28-9e156d12c3ea.png)

**Model yang dikembangkan**
![newplot (2)](https://user-images.githubusercontent.com/41296422/137366585-2cff2287-6756-48c2-a670-fb9a1434631d.png)

Secara kasat mata, dari kedua model tersebut dapat memprediksi data uji dengan baik, akan tetapi kita harus memilih model manakah yang terbaik dengan cara mencari model dengan nilai _error rate_ terkecil.

## Evaluating
Pada proyek ini, model yang dibuat merupakan kasus regresi dan menggunakan metriks perhitungan _Root Mean Squared Error_ (RMSE) dan _Mean Absolute Error_ (MAE). Penggunaan metriks tersebut karena memberikan bobot yang relatif tinggi untuk kesalahan besar. Berikut adalah rumus dari perhitungan RMSE dan MAE:

**RMSE**
<img src="https://user-images.githubusercontent.com/16319829/81180309-2b51f000-8fee-11ea-8a78-ddfe8c3412a7.png" width="150" height="280">

![1_lqDsPkfXPGen32Uem1PTNg](https://user-images.githubusercontent.com/41296422/137367800-2dc7bd32-f39e-447e-915f-c623a5192a70.png)

Nilai RMSE didapatkan dari perhitungan jumlah setiap nilai prediksi dikurangi nilai asli dipangkat dua lalu dibagikan dengan banyaknya data dan terakhir diakarkan.

**MAE**
![1_OVlFLnMwHDx08PHzqlBDag](https://user-images.githubusercontent.com/41296422/137367898-3f9c131d-a300-4c70-a9bc-8e7d8d1458ee.gif)

Nilai MAE didapatkan dari perhitungan jumlah setiap nilai asli dikurangi nilai prediksi dipangkat dua lalu dibagikan dengan banyaknya data, dan nilai tersebut mutlak (bukan negatif).

Pada tabel di bawah ini adalah hasil dari perhitungan RMSE dan MAE dari kedua model di atas.

||Root Mean Squared Error|Mean Absolute Error|
|------|----------|-------|
|Model _Baseline_|0.002242|0.001706|
|Model yang dikembangkan|0.001476|0.001147|

Dapat disimpulkan bahwa, model yang dikembangkan lebih baik daripada model _baseline_.
