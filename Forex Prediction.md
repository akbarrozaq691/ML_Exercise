# Laporan Proyek Machine Learning - Hasri Akbar Awal Rozaq

## Domain Proyek
Domain proyek yang dipilih dalam proyek *machine learning* ini adalah mengenai **keuangan** dengan judul proyek "Prediksi Nilai Tukar Uang EUR terhadap USD".

*Foreign exchange* (Forex) adalah pasar yang mengkhususkan diri dalam perdagangan pertukaran valuta asing [[1]](https://ojs.unud.ac.id/index.php/bse/article/view/2195). Dalam pasar *forex* sendiri, seseorang yang melakukan perdagangan biasa disebut dengan *trader* dimana dapat menentukan antara dua pilihan / posisi, baik membeli atau menjual mata uang yang diperdagangkan. Jika kurs jual mata uang lebih besar dari tingkat pembelian, maka menghasilkan keuntungan bagi orang tersebut [[2]](https://www.researchgate.net/publication/343342034). Nilai tukar mata uang yang sering tidak stabil menjadikan bisnis *forex* sebagai bisnis beresiko tinggi tetapi juga sebagai bisnis dengan keuntungan besar pula sehingga nilai pertukaran yang dialami perlu diperhatikan. Permasalahan pada proses tersebut muncul yaitu diperlukan proses prediksi yang tepat untuk meminimalkan resiko dan meningkatkan sebuah keuntungan [[3]](https://jurnal.umk.ac.id/index.php/simet/article/view/453).

Adanya kecanggihan teknologi saat ini sangat membantu kinerja manusia. Pada proyek kali ini, saya akan membangun sebuah model *machine learning* yang diharapkan dapat memprediksi data nilai tukar mata uang EUR (Euro) terhadap USD (US Dollar). Model yang dibuat menggunakan teknik *predictive modelling* dimana sangat cocok digunakan untuk diterapkan pada kasus proyek ini. Selain itu, model ini juga diharapkan bisa diimplementasikan pada beberapa *platform* yang populer seperti *web* ataupun *android*.

## Business Understanding
### Problem Statements
Berdasarkan kondisi yang telah diuraikan sebelumnya, berikut ini merupakan rincian masalah yang dapat diselesaikan:
- Bagaimana cara membuat model *machine learning* untuk memprediksi data *forex*?
- Model manakah yang terbaik untuk mengatasi masalah tersebut?

### Goals
Untuk menjawab pertanyaan pada *problem statements* di atas, berikut tujuan dari dibuatnya proyek ini:
- Model yang dipilih dapat memprediksi nilai tukar mata uang dengan rentang waktu tertentu
- Dapat menemukan model terbaik terhadap data forex dengan cara membandingkan nilai *error rate* terkecil dari model yang ada

### Solutions Statements
Solusi yang dilakukan untuk memenuhi tujuan dari proyek ini di antaranya:
- Untuk pra-pemrosesan data dapat dilakukan beberapa teknik, antara lain:
  - Menghapus data yang kosong
  - Melakukan **pembagian** dataset menjadi dua bagian dengan persentase 80% untuk data latih dan 20% untuk data uji
  - Melakukan **standarisasi data** pada fitur data
  - Karena data bersifat *time-series*, maka alangkah lebih baik diubah menjadi data sekuensial menggunakan TimeseriesGenerator
  
  Poin pra-pemrosesan data akan dijelaskan secara rinci pada bagian `Data Preparation`.
- Untuk pembuatan model sendiri menggunakan algoritma **LSTM (*Long-Short Term Memory*)** sebagai model *baseline*. Algoritma tersebut dipilih karena mudah diimplementasikan dan juga cocok untuk kasus data sekuensial (*time series*). LSTM sendiri memiliki tiga *gate*, yaitu *input gate*, *forget gate*, dan *output gate* [[4]](https://ieeexplore.ieee.org/document/8125846). Algoritma ini dapat mengingat masa lalu untuk memprediksi masa depan melalui *gate* pengingat. Cara kerja algoritma ini adalah sebagai berikut (diterjemahkan dari [[5]](https://ntnuopen.ntnu.no/ntnu-xmlui/handle/11250/2559922)):
  - Data akan masuk ke dalam *input gate* terlebih dahulu
  - Data yang telah masuk akan dipelajari pada *short term memory* 
  - Pada ingatan lama (*long term memory*), data yang tidak berguna akan dilupakan
  - Untuk mengingat kembali, ingatan lama yang belum dilupakan ditambahkan dengan data yang sedang dipelajari akan menjadi *long term memory* yang baru
  - Untuk memprediksi, ingatan lama yang belum dilupakan ditambahkan dengan data yang sedang dipelajari dan akan digunakan, lalu menjadi *short term memory* yang baru
  - Data yang diprediksi akan dikeluarkan pada *output gate*

  Selain itu, berikut ini adalah kelebihan dan kelemahan algoritma LSTM:
  - Kelebihan (diterjemahkan dari [[6]](https://www.springer.com/gp/book/9789811026652)):
    - Adanya arsitektur mengingat dan melupakan output yang akan diproses kembali menjadi input
    - Dapat mempertahankan error yang terjadi ketika melakukan backpropagation sehingga tidak memungkinkan kesalahan meningkat
  - Kekurangan (diterjemahkan dari [[7]](https://arxiv.org/abs/1803.04831)):
    - Memiliki arsitektur yang kompleks sehingga beban komputasi menjadi tinggi terutama ketika diterapkan pada kasus skala besar
- Kemudian model *baseline* tersebut dikembangkan dengan pengaturan *hyperparameter* dengan cara membuat *custom loss function*, menambahkan *layer* LSTM, dan penerapan *learning rate* pada *optimizer function* dimana dengan memodifikasi parameter tersebut, dapat mengurangi nilai *error*. Penggunaan *custom loss function* diambil dari metode yang bernama *Huber*. Berikut adalah rumus dari *Huber Loss Function*

    ![Capture](https://user-images.githubusercontent.com/41296422/137327705-7799a336-9a43-4d24-9c9e-660b137d8fa0.JPG)

    Cara kerja dari metode tersebut antara lain:
     - Menghitung nilai *error* terlebih dahulu dimana nilai tersebut didapatkan dari pengurangan antara data asli dan data prediksi
     - Menghitung nilai *loss* terkecil dengan cara nilai *error* pangkat 2 dibagi 2
     - Menghitung nilai *loss* terbesar dengan cara nilai *thershold* dikali dengan nilai *error* mutlak dikurangi 1/2 dari nilai *threshold*
     - Lalu akan dicari nilai *loss* sesuai *thershold*
  Dengan adanya pengaturan *hyperparameter* ini harapannya akan menciptakan model yang lebih akurat dan memiliki nilai *error rate* kecil. 
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

Pada berkas yang diunduh yakni `EURUSD=X.csv` berisi informasi metriks nilai tukar uang EUR/USD dengan jumlah 4663 data. Terdapat 6 buah data numerik (tipe data float64) dan 1 buah data *date time* (tipe data datetime64). Dataset tersebut memiliki data kosong kecuali pada kolom tanggal. Untuk mengenal variabel apa saja pada dataset tersebut, dapat dilihat pada poin-poin sebagai berikut:
1. `Date`: Tanggal dimana menunjukkan waktu terjadinya pembukaan dan penutup harga, pada dataset kali ini, data tersebut berisi waktu harian dan sangat penting untuk dianalisis apakah harga naik / turun dalam satu hari terakhir
2. `Open`: Harga pertama kali transaksi dilakukan pada hari itu. Harga *open* tersebut mencerminkan semua informasi pasar yang ada, yang terjadi atau muncul diantara harga penutupan sehari sebelumnya dan ketika saat-saat terakhir pemodal boleh memasukkan order ke mesin bursa.
3. `High`: Kisaran harga pergerakan harian dari saham tersebut dimana pemodal memiliki keberanian atau rasionalitas untuk melakukan posisi beli.
4. `Low`: Kisaran harga pergerakan harian dari saham tersebut dimana pemodal memiliki keberanian atau rasionalitas untuk melakukan posisi jual.
5. `Close`: Harga close ini mencerminkan semua informasi yang ada pada semua pelaku pasar (terutama pelaku pasar institusi yang memiliki informasi yang lebih akurat) pada saat perdagangan saham tersebut berakhir.
6. `Adjusted Close`: Seperti halnya variabel close, variabel adjust close merupakan harga penutupan yang disesuaikan.
7. `Volume`: Representasi dari aktivitas yang berlangsung selama suatu periode trading.

Pada kasus kali ini yang akan diterapkan adalah variabel `Date` dan `Close` karena keduanya bisa memicu signal beli atau signal jual untuk waktu yang akan datang.
Kemudian terdapat juga visualisasi data untuk kolom `Close` dengan `Date` sebagai indeksnya:
![newplot (1)](https://user-images.githubusercontent.com/41296422/137333358-6fb353be-5227-4e1e-9895-881e86f51f3b.png)

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
