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
     - Menghitung nilai _loss_ terbesar dengan cara nilai _thershold_ dikali dengan nilai _error_ mutlaik dikurangi 1/2 dari nilai _threshold_
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

Pada berkas yang diunduh yakni `EURUSD=X.csv`
