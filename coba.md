# Laporan Proyek Machine Learning - Hasri Akbar Awal Rozaq

## Domain Proyek
Domain proyek yang dipilih dalam proyek *machine learning* ini adalah mengenai **keuangan** dengan judul proyek "Prediksi Nilai Tukar Uang EUR terhadap USD".

*Foreign exchange* (Forex) adalah pasar yang mengkhususkan diri dalam perdagangan pertukaran valuta asing. Dalam pasar *forex* sendiri seseorang yang melakukan perdagangan biasa disebut dengan *trader* memiliki dua pilihan, baik membeli atau menjual mata uang yang diperdagangkan. Jika kurs jual mata uang lebih besar dari tingkat pembelian, itu menghasilkan keuntungan bagi orang tersebut [(Islam et.al., 2020)](https://www.researchgate.net/publication/343342034). Nilai tukar mata uang yang sering tidak stabil menjadikan bisnis *forex* sebagai bisnis beresiko tinggi tetapi juga sebagai bisnis dengan keuntungan besar pula sehingga nilai pertukaran yang dialami perlu diperhatikan. Dari aspek tersebut maka diperlukan proses prediksi yang tepat untuk meminimalkan resiko dan meningkatkan sebuah keuntungan [(Kusumodestoni, 2015)](https://jurnal.umk.ac.id/index.php/simet/article/view/453). Data tersebut bersifat *time-series* (sekuensial) sehingga perlu algoritma yang dapat mengatasi masalah tersebut. Sehingga, teknik *predictive modelling* sangat cocok digunakan untuk diterapkan pada kasus tersebut.

## Business Understanding
### Problem Statements
Berdasarkan kondisi yang telah diuraikan sebelumnya, perancang algoritma ingin membantu *trader* dalam mengembangkan sistem prediksi *forex* untuk menjawab permasalahan tersebut.
* Bagaimana cara kerja algoritma *time-series* memprediksi data forex?
* Algoritma manakah yang terbaik untuk mengatasi masalah tersebut?
* Berapa nilai tukar uang dengan waktu tertentu?

### Goals
Untuk menjawab pertanyaan tersebut, saya akan membuat predictive modelling dengan *goals* sebagai berikut:
* Algoritma yang dipilih dapat memprediksi nilai tukar mata uang dengan rentang waktu tertentu.
* Dapat menemukan algoritma terbaik terhadap data forex dengan membandingkan nilai error rate terkecil dari model yang ada.

### Solutions Statements
Solusi yang diberikan adalah menggunakan algoritma *time-series* yaitu LSTM (*Long-Short Term Memory*) dan SARIMA (*Seasonal Auto Regressive Moving Average*).
- **Long-Short Term Memory**
LSTM merupakan salah satu jenis dari *Recurrent Neural Network* (RNN) dimana modifikasi dilakukan pada RNN dengan menambahkan sel memori untuk menyimpan informasi dalam waktu yang lama, pernyataan tersebut juga bisa dikatakan kelebihan dari LSTM. LSTM diusulkan sebagai solusi untuk mengatasi terjadinya *vanishing gradient* pada RNN saat memproses data sequential yang panjang. Berikut adalah proses LSTM secara umum:
*Forget Gate* => *Input Gate* => *Cell Gate* => *Output Gate*
Langkah pertama dimulai dari *forget gate* dimana informasi yang kurang dibutuhkan akan dihilangkan. Lalu, informasi yang penting akan masuk ke *input gate* untuk dipilah dan ditentukan informasinya yang akan diperbaharui ke *cell gate*. Setelah itu memperbaharui *cell gate* dari *cell gate* lama. Terakhir, informasi yang telah diolah akan dioutputkan.

- **Seasonal Auto Regressive Moving Average**
SARIMA adalah metode peramalan *time series* untuk model data fluktuatif dengan pola data musiman. Kelebihan metode SARIMA ada pada pembelajaran yang cepat dan pemilihan model yang tepat yang mempunyai pola musiman. 
## Data Understanding
