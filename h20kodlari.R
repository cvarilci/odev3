# Genel bir uygun model araştırımı
# h2o.automl gerçekten de R'daki en keyifli ve güçlü araçlardan biri. 
# Python'daki lazypredict'in "tek satırda sonuç al" felsefesini alır ve 
# bunu endüstriyel seviyede bir güçle birleştirir.

# R Projesi - Adım 3: Hızlı Model Karşılaştırması (h2o.automl ile)
# h2o Java tabanlı, bellek-içi (in-memory), dağıtık ve ölçeklenebilir bir makine öğrenmesi platformudur. 
# R (ve Python) sadece bu güçlü platforma komut gönderen bir arayüz görevi görür.

# 1. H2O Paketinin Kurulması ve Başlatılması
# h2o kurulumu diğer R paketlerinden biraz farklıdır, çünkü aynı zamanda bir 
# Java (.jar) dosyasına da ihtiyaç duyar.
# Aşağıdaki komutlar ile bu durum gerçekleştirilir.
# Eğer kurulu değilse, h2o paketini kuralım
if (!require(h2o)) {
  install.packages("h2o")
}

# h2o kütüphanesini yükle
library(h2o)

# H2O kümesini (cluster) başlat
# nthreads = -1: Bilgisayardaki tüm çekirdekleri kullan
# max_mem_size: H2O'ya ne kadar RAM kullanabileceğini belirtir. 
#               Daha büyük veri setleri için artırılabilir.
h2o.init(nthreads = -1, max_mem_size = "4g")

# Veriyi H2O Formatına Çevirme
# h2o kendi optimize edilmiş veri formatını (H2OFrame) kullanır. 
# tidymodels ile böldüğümüz eğitim ve test setlerini bu formata çevirmemiz gerekiyor.

# Önemli Not: h2o kendi içinde kategorik değişkenleri çok verimli bir şekilde işleyebilir. 
# Bu nedenle, ona tidymodels'ın recipe'i ile ön işlemden geçirilmiş veriyi değil, 
# orijinal, ham train_data ve test_data'yı vereceğiz. 
# Bu, h2o'nun kendi içindeki optimizasyonlardan tam olarak faydalanmasını sağlar.

# Eğitim ve test veri setlerini H2OFrame'e dönüştür
h2o_train <- as.h2o(train_data)
h2o_test  <- as.h2o(test_data)

# Hedef değişkenin ve öngörücülerin adlarını tanımla
# Orijinal 'charges' sütununu değil, logaritmik olanı kullanacağız.
y <- "log_charges"
x <- setdiff(names(h2o_train), c(y, "charges")) # 'log_charges' ve 'charges' dışındaki her şey

# AutoML'i Çalıştırma
# h2o.automl() fonksiyonunu sadece birkaç parametre ile çalıştıracağız.

# AutoML sürecini başlat
automl_models <- h2o.automl(
  x = x,
  y = y,
  training_frame = h2o_train,
  validation_frame = h2o_test, # Performansı test seti üzerinde doğrula
  max_runtime_secs = 120,      # Toplamda 2 dakika (120 saniye) çalışsın
  seed = 42                    # Tekrarlanabilirlik için
)

# Açıklamalar:

# training_frame ve validation_frame: Modeli eğitmek ve performansını ölçmek için kullanılacak veri setleri.
# max_runtime_secs: AutoML'in ne kadar süreyle çalışacağını belirler. 
# Süre ne kadar uzun olursa, o kadar fazla model dener ve o kadar iyi hiperparametreler bulabilir. 
# Küçük bir veri seti için 1-2 dakika genellikle yeterlidir.
# h2o.automl arka planda şunları yapar:
# Farklı model türlerini (Gradient Boosting (GBM), XGBoost, Random Forest, Deep Learning, GLM vb.) dener.
# Her model için hiperparametre optimizasyonu yapar.
# Bu modelleri birleştirerek "Stacked Ensemble" adı verilen, 
# genellikle tekil modellerden daha güçlü olan meta-modeller oluşturur.

# Sonuçları İnceleme: Liderlik Tablosu
# AutoML süreci bittiğinde, sonuçları içeren bir "liderlik tablosu" (leaderboard) elde ederiz. 
# Bu, lazypredict'in çıktısına çok benzer, ancak çok daha güvenilirdir.

# Liderlik tablosunu al
# Liderlik tablosunu al
leaderboard <- automl_models@leaderboard 

# Liderlik tablosunu yazdır. En iyi model en üstte olacak.
# Metrikler (rmse, mae, r2) test verisi üzerinde hesaplanmıştır.
print(leaderboard, n = 10) # En iyi 10 modeli göster

# İsterseniz en iyi modeli de bir değişkene atayabilirsiniz
best_model <- automl_models@leader

# En iyi modelin detaylı özetini görmek için
print(best_model)

# Çıktıyı Yorumlama:

# model_id: Her bir modelin benzersiz kimliği.
# rmse: Kök Ortalama Kare Hata (Düşük olan daha iyi).
# mae: Ortalama Mutlak Hata (Düşük olan daha iyi).
# r2: R-Kare Skoru (Yüksek olan daha iyi).

# H2O Kümesini Kapatma (Önemli)
# İşiniz bittiğinde, H2O'nun arkaplanda RAM kullanmaya devam etmemesi için kümesini kapatmak iyi bir pratiktir.

h2o.shutdown(prompt = FALSE)







