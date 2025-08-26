#install.packages("tidyverse")
#install.packages("DataExplorer")

# Gerekli kütüphaneleri yükle
library(tidyverse)# ggplot2, dplyr, readr gibi paketleri içerir
library(DataExplorer)

# Veri setini yükle
df <- read_csv("insurance.csv")

# Verinin ilk birkaç satırına bak (Python: df.head())
head(df)

# Verinin yapısına genel bir bakış (Python: df.info())
glimpse(df)

# Sayısal değişkenlerin istatistiksel özeti (Python: df.describe())
summary(df)

# Veri setine dair otomatik bir EDA raporu oluştur
create_report(df)

# Kategorik Değişkenler vs. charges
# Sigara içme durumuna göre masraflar (Python: sns.boxplot)
ggplot(df, aes(x = smoker, y = charges, fill = smoker)) +
  geom_boxplot() +
  labs(title = "Sigara İçme Durumuna Göre Tıbbi Masraflar",
       x = "Sigara İçme Durumu",
       y = "Masraflar") +
  theme_minimal()

# Cinsiyete göre masraflar
ggplot(df, aes(x = sex, y = charges, fill = sex)) +
  geom_boxplot() +
  labs(title = "Cinsiyete Göre Tıbbi Masraflar") +
  theme_minimal()

# Sayısal Değişkenler vs. charges
# Yaş ve masraflar arasındaki ilişki (Python: sns.scatterplot)
ggplot(df, aes(x = age, y = charges, color = smoker)) +
  geom_point() + # Saçılım grafiği için geom_point kullanılır
  labs(title = "Yaş ve Sigara Durumuna Göre Masraflar",
       x = "Yaş",
       y = "Masraflar") +
  theme_minimal()

# BMI ve masraflar arasındaki ilişki
ggplot(df, aes(x = bmi, y = charges, color = smoker)) +
  geom_point() +
  labs(title = "BMI ve Sigara Durumuna Göre Masraflar") +
  theme_minimal()

# Hedef Değişkenin Dağılımı ve Log Dönüşümü
# 'charges' sütununun dağılımı (Python: sns.histplot)
p1 <- ggplot(df, aes(x = charges)) +
  geom_histogram(bins = 50, fill = "skyblue", color = "white") +
  labs(title = "Dönüşüm Öncesi Orijinal Dağılım")

# Log dönüşümü uygulanmış yeni bir sütun ekle (Python: df['log_charges'] = ...)
# R'da bu işlem için dplyr'ın mutate() fonksiyonu çok kullanışlıdır.
df <- df %>%
  mutate(log_charges = log1p(charges))

# Dönüştürülmüş dağılım
p2 <- ggplot(df, aes(x = log_charges)) +
  geom_histogram(bins = 50, fill = "salmon", color = "white") +
  labs(title = "Logaritmik Dönüşüm Sonrası Dağılım")

# İki grafiği yan yana göstermek için (patchwork paketi gerekir)
# install.packages("patchwork")
library(patchwork)
p1 + p2

# R Projesi - Adım 2: Veri Ön İşleme ve Hazırlık (tidymodels ile)
# Bu adım, Python'da get_dummies, train_test_split ve StandardScaler ile yaptığımız işlerin R'daki karşılığıdır. 
# tidymodels bu adımları "tarifler" (recipes) ve "bölmeler" (splits) olarak adlandırılan iki ana konseptle yönetir.

# Gerekli Paketlerin Kurulması ve Yüklenmesi
# Eğer yüklü değilse, öncelikle tidymodels paketini kuralım. 
# Bu, rsample, recipes, parsnip gibi birçok alt paketi de beraberinde getirir.

#install.packages("tidymodels")

# tidymodels meta-paketini yükle
library(tidymodels)

# Bir önceki adımdan gelen df veri setini kullanmaya devam ediyoruz.
# log_charges sütununu eklemiştik.

# 2. Veriyi Eğitim ve Test Olarak Bölme (rsample ile)
# Python'daki train_test_split'in tidymodels'daki karşılığı initial_split fonksiyonudur.

# 1. Veriyi bölmek için bir "split" nesnesi oluştur
# Veriyi %80 eğitim, %20 test olarak ayıracağız.
# 'strata = log_charges' kullanmak, hedef değişkenin dağılımının
# hem eğitim hem de test setinde benzer olmasını sağlar. Bu iyi bir pratiktir.
set.seed(42) # Tekrarlanabilirliği sağlamak için (Python'daki random_state gibi)
data_split <- initial_split(df, prop = 0.80, strata = log_charges)

# 2. Split nesnesinden eğitim ve test setlerini çıkar
train_data <- training(data_split)
test_data  <- testing(data_split)

# Bölme işleminin doğruluğunu kontrol et
cat("Eğitim seti boyutu:", nrow(train_data), "\n")
cat("Test seti boyutu:", nrow(test_data), "\n")

# 3. Veri Ön İşleme "Tarifi" Oluşturma (recipes ile)
# Bu, tidymodels'ın en güçlü ve en önemli konseptidir. Bir "tarif" (recipe), 
# modelinize girmeden önce verilere uygulanacak tüm ön işleme adımlarını 
# (one-hot encoding, ölçeklendirme, eksik veri doldurma vb.) tanımladığınız bir plandır.

# Python'da bu adımları ayrı ayrı yaparken, recipes tüm adımları tek bir nesnede birleştirir.

# 1. Bir "tarif" nesnesi oluştur
insurance_recipe <- recipe(log_charges ~ ., data = train_data) %>%
  # Adım 1: Gereksiz olan orijinal 'charges' sütununu çıkar
  step_rm(charges) %>%
  # Adım 2: Tüm nominal (kategorik) değişkenlere one-hot encoding uygula
  # Bu, 'sex', 'smoker' ve 'region' sütunlarını otomatik olarak bulup
  # 0/1'lerden oluşan yeni sütunlara dönüştürecektir (Python: pd.get_dummies)
  step_dummy(all_nominal_predictors()) %>%
  # Adım 3: Tüm sayısal öngörücüleri (predictors) merkezileştir ve ölçeklendir
  # Bu, 'age', 'bmi', 'children' ve yeni oluşan dummy değişkenleri
  # otomatik olarak bulup standartlaştıracaktır (Python: StandardScaler)
  step_normalize(all_numeric_predictors())

# Tarif nesnesinin özetini görüntüle
summary(insurance_recipe)

# Ek bilgi
# Dikkat bu bilgiler sadece açıklayıcı bilgilerdir.
# Tarif nesnesinin özetinde değişkenlerin hepsinin neden chr şeklinde gözüktüğü
# üzere sorulan sorunun cevabı ve aslında ne olduğu hakkındaki bilgidir.
# Analiz ile ilgilisi olmayıp açıklayıcı bir anlatım ve bunu göstermeye yarayan
# kodlardır.

# summary(insurance_recipe) komutunun çıktısındaki type sütununda her şeyin chr (character) 
# olarak görünmesinin nedeni, bu özetin size verinin o anki tipini değil, tarife kaydedilmiş olan orijinal tipini göstermesidir.

# Daha basit bir ifadeyle:
  
# Tarifi Oluşturma Anı: recipe(log_charges ~ ., data = train_data) komutunu çalıştırdığınızda, 
# tarif (insurance_recipe) eğitim verisine (train_data) bakar. 
# Bu aşamada, age, bmi gibi sütunlar numeric (sayısal), sex, smoker gibi 
# sütunlar ise character (kategorik) olarak tanımlıdır. summary() komutu, bu ilk ve ham bilgiyi size raporlar.
# Adımların "Planlanması": step_dummy(all_nominal_predictors()) ve 
# step_normalize(all_numeric_predictors()) gibi adımları eklediğinizde, 
# henüz veriye hiçbir işlem yapmadınız. Sadece tarife, "ileride bu veriyi işlerken, 
# kategorik olanları dummy'ye çevir ve sayısal olanları normalize et" diye bir talimat listesi eklediniz.
# Tarif tembeldir; siz ona "şimdi bu tarifi hazırla ve uygula" demeden hiçbir şeyi değiştirmez. 
# Bu yüzden summary() komutu size hâlâ verinin orijinal, işlenmemiş halindeki tiplerini gösteriyor.

# Peki, Değişiklikleri Nasıl Görebiliriz?
# Tarifin uygulanmaya hazır hale geldiğinde ve uygulandığında neler olacağını görmek için 
# prep() ve bake() fonksiyonlarını kullanırız.

# prep(recipe): Bu komut, tarifi eğitim verisi üzerinde eğitir. 
# Yani, step_normalize için gereken ortalama ve standart sapma gibi istatistikleri train_data'dan hesaplar, 
# step_dummy'nin hangi kategorileri yeni sütunlara çevireceğini belirler.
# bake(prepped_recipe, new_data = ...): Bu komut, eğitilmiş tarifi yeni bir veri setine uygular.
# Bu süreci test etmek için aşağıdaki kodu çalıştırabilirsiniz. 
# Bu, bir sonraki modelleme adımında arka planda otomatik olarak yapılacak olan işlemin manuel bir simülasyonudur.

# 1. Tarifi eğitim verisi üzerinde "hazırla" (eğit)
prepped_recipe <- prep(insurance_recipe, training = train_data)

# 2. Hazırlanmış tarifi eğitim verisine "uygula"
#    ve sonucun ilk birkaç satırına bak
baked_train_data <- bake(prepped_recipe, new_data = train_data)
head(baked_train_data)

# 3. İŞTE ASIL KANIT: "bake" edilmiş verinin yapısına bak
#    Artık tüm sütunların sayısal (dbl - double) olduğunu göreceksiniz.
glimpse(baked_train_data)

#----------------AÇIKLAMA SONU-------------------------------------------

# Analize devam
# R Projesi - Adım 4.1: Lineer Model Ailesini Kurma (tidymodels ile)
# tidymodels'da modelleme süreci genellikle şu 3 adımdan oluşur:

# Model Tanımlama (parsnip): Ne tür bir model kullanmak istediğimizi ve hangi motoru (yani R paketini) kullanacağımızı belirtiriz.
# İş Akışı Oluşturma (workflows): Daha önce hazırladığımız recipe ile bu model tanımını birleştiririz.
# Modeli Eğitme ve Değerlendirme (fit, predict): İş akışını eğitim verisiyle eğitir ve test verisi üzerinde performansını ölçeriz.
# 1. Gerekli Kütüphaneler
# tidymodels zaten yüklü. Lineer modeller için glmnet paketi genellikle arka planda kullanılır ve 
# tidymodels onu otomatik olarak çağırabilir.

# tidymodels zaten yüklü olmalı
library(tidymodels)

# Bir önceki adımdaki `train_data`, `test_data` ve `insurance_recipe`'i kullanacağız
# 1. Model Tanımlama: Lineer regresyon modeli kullanacağımızı belirtiyoruz.
#    'lm' motoru, standart lineer modeller için R'ın temel motorudur.
lm_model <- linear_reg() %>%
  set_engine("lm")

# 2. İş Akışı (Workflow) Oluşturma: Tarifi ve modeli birleştiriyoruz.
lm_workflow <- workflow() %>%
  add_recipe(insurance_recipe) %>%
  add_model(lm_model)

# 3. Modeli Eğitme: İş akışını eğitim verisiyle çalıştırıyoruz.
#    .fit() fonksiyonu, tarifi verilere uygular ve ardından modeli eğitir.
lm_fit <- lm_workflow %>%
  fit(data = train_data)

# 'lm_fit' artık eğitilmiş bir model nesnesidir.
# Örneğin, modelin katsayılarını görmek için:
# tidy(lm_fit)

# Ridge, Lasso ve Elastic Net Regresyon
# Bu üç model, aynı temel model tanımını kullanır: linear_reg(). 
# Aralarındaki fark, penalty (ceza) ve mixture (karışım) parametreleriyle belirlenir.

# Ridge: mixture = 0
# Lasso: mixture = 1
# Elastic Net: mixture 0 ile 1 arasında bir değer alır.
# Bu modeller için glmnet motorunu kullanacağız.

# Ön Açıklama 
# 1. Model Tanımlama: Regülarizasyonlu bir lineer model.
#    penalty ve mixture parametrelerinin optimize edileceğini 'tune()' ile belirtiyoruz.
#    Şimdilik manuel olarak değerler atayacağız, tıpkı Python'daki gibi.
#glmnet_model <- linear_reg(penalty = 0.01, mixture = 1) %>% # Bu bir Lasso modeli (mixture=1)
#  set_engine("glmnet")

# Ridge ve Elastic Net için de benzer tanımlamalar yapabiliriz.
# Hadi tümünü tek bir yerde yapıp sonuçları toplayalım.

# Tüm Lineer Modelleri Eğitme ve Sonuçları Karşılaştırma
# Sonuçları tek bir tabloda toplamak için R'da bir tibble (modern bir data frame) oluşturalım.

# Sonuçları saklamak için boş bir tibble oluştur
linear_results <- tibble()

# --- Basit Lineer Regresyon ---
# (Yukarıda eğittik, şimdi sadece tahmin ve değerlendirme yapacağız)
lm_predictions <- predict(lm_fit, new_data = test_data) %>%
  # Tahminleri orijinal test verisiyle birleştir
  bind_cols(test_data)

# --- Ridge Regresyon (mixture = 0) ---
ridge_model <- linear_reg(penalty = 0.01, mixture = 0) %>% set_engine("glmnet")
ridge_workflow <- workflow() %>% add_recipe(insurance_recipe) %>% add_model(ridge_model)
ridge_fit <- fit(ridge_workflow, data = train_data)
ridge_predictions <- predict(ridge_fit, new_data = test_data) %>%
  bind_cols(test_data)

# --- Lasso Regresyon (mixture = 1) ---
lasso_model <- linear_reg(penalty = 0.01, mixture = 1) %>% set_engine("glmnet")
lasso_workflow <- workflow() %>% add_recipe(insurance_recipe) %>% add_model(lasso_model)
lasso_fit <- fit(lasso_workflow, data = train_data)
lasso_predictions <- predict(lasso_fit, new_data = test_data) %>%
  bind_cols(test_data)

# --- Elastic Net (mixture = 0.5) ---
elastic_net_model <- linear_reg(penalty = 0.01, mixture = 0.5) %>% set_engine("glmnet")
elastic_net_workflow <- workflow() %>% add_recipe(insurance_recipe) %>% add_model(elastic_net_model)
elastic_net_fit <- fit(elastic_net_workflow, data = train_data)
elastic_net_predictions <- predict(elastic_net_fit, new_data = test_data) %>%
  bind_cols(test_data)

# --- Metrikleri Hesaplama ve Birleştirme ---

# Metrikleri hesaplamak için bir fonksiyon oluşturalım (tekrarı önlemek için)
calculate_metrics <- function(predictions, model_name) {
  predictions %>%
    # ÖNEMLİ: Logaritmik tahminleri ve gerçek değerleri orijinal ölçeğe geri çevir
    mutate(
      truth_original = expm1(log_charges),
      pred_original = expm1(.pred)
    ) %>%
    # Metrikleri hesapla (yardstick paketi ile)
    metrics(truth = truth_original, estimate = pred_original) %>%
    # Sonucu düzenle
    mutate(model = model_name) %>%
    select(model, .metric, .estimate)
}

# Her model için metrikleri hesapla ve birleştir
final_linear_results <- bind_rows(
  calculate_metrics(lm_predictions, "Linear Regression"),
  calculate_metrics(ridge_predictions, "Ridge Regression"),
  calculate_metrics(lasso_predictions, "Lasso Regression"),
  calculate_metrics(elastic_net_predictions, "Elastic Net")
)

# Sonuçları daha okunaklı bir formata getirelim
final_linear_results %>%
  pivot_wider(names_from = .metric, values_from = .estimate)

# şimdi sıra SVR modelde
# R Projesi - SVR ve Hiperparametre Optimizasyonu (tune ile)
# tidymodels'da optimizasyon süreci şu adımları izler:

# Yeniden Örnekleme Planı Oluşturma (vfold_cv): Güvenilir sonuçlar için veriyi K-katlı çapraz doğrulama (K-Fold Cross-Validation) setlerine böleriz.
# Model Tanımlama (parsnip): Optimize etmek istediğimiz parametreleri tune() fonksiyonu ile işaretleriz.
# Optimizasyon Izgarası (grid_regular): Denemek istediğimiz hiperparametre kombinasyonlarını içeren bir "ızgara" (grid) oluştururuz.
# Optimizasyonu Çalıştırma (tune_grid): Modeli, tarifi ve yeniden örnekleme verisini birleştirerek tüm parametre kombinasyonlarını deneriz.
# En İyi Sonuçları Değerlendirme: En iyi performansı gösteren parametre setini seçer ve bu parametrelerle final modelimizi eğitiriz.
# Gerekli Kütüphaneler ve Hazırlık
# tidymodels zaten yüklü
library(tidymodels)
install.packages("kernlab")

# Bir önceki adımdaki `train_data`, `test_data` ve `insurance_recipe`'i kullanıyoruz.
# Çapraz Doğrulama (Cross-Validation) Setleri Oluşturma
# Model performansını daha sağlam bir şekilde ölçmek için eğitim verimizi 5 parçaya böleceğiz
set.seed(42)
data_folds <- vfold_cv(train_data, v = 5, strata = log_charges)
# SVR Modelini Optimizasyon İçin Tanımlama
# SVR modelini (svm_rbf) tanımlarken, Python'da optimize ettiğimiz C ve gamma parametrelerini tune() olarak işaretleyeceğiz. 
# R'da C parametresi cost olarak adlandırılır. gamma ise rbf_sigma'nın tersiyle ilişkilidir, 
# bu yüzden biz doğrudan rbf_sigma'yı optimize edeceğiz.
# 1. Model Tanımlama: 'rbf' çekirdekli bir SVR.
#    'cost' (C) ve 'rbf_sigma' (gamma ile ilgili) parametrelerini
#    optimize edeceğimizi belirtiyoruz.
svr_model <- svm_rbf(cost = tune(), rbf_sigma = tune()) %>%
  set_engine("kernlab") %>% # SVR için popüler bir R motoru
  set_mode("regression")
# İş Akışı (Workflow) Oluşturma
# Artık modelimizi bir iş akışına ekleyebiliriz.
# İş akışı: Tarifi ve optimize edilecek modeli birleştir.
svr_workflow <- workflow() %>%
  add_recipe(insurance_recipe) %>%
  add_model(svr_model)
# Hiperparametre Optimizasyonunu Çalıştırma
# Şimdi tune_grid fonksiyonu ile tüm süreci birleştireceğiz. 
# Bu fonksiyon, belirttiğimiz tüm cost ve rbf_sigma kombinasyonlarını 5 katlı 
# çapraz doğrulama setlerinin her birinde deneyerek en iyi sonucu bulacak.

# Optimizasyonu başlat
print("SVR için hiperparametre optimizasyonu başlıyor... (Bu işlem biraz zaman alabilir)")
start_time <- Sys.time()

# tune_grid'i çalıştır
svr_tune_results <- tune_grid(
  svr_workflow,           # İş akışımız
  resamples = data_folds, # Çapraz doğrulama verimiz
  grid = 20               # 20 farklı hiperparametre kombinasyonu dene
)

end_time <- Sys.time()
print(paste("Optimizasyon", round(end_time - start_time, 2), "saniyede tamamlandı."))

# Optimizasyon sonuçlarını göster
# En iyi performansı (en düşük rmse) gösteren parametreleri listeleyecek
show_best(svr_tune_results, metric = "rmse")

# Final Modeli Eğitme ve Test Verisi Üzerinde Değerlendirme
# tune_grid en iyi parametreleri buldu. Şimdi bu parametreleri iş akışımıza ekleyip, 
# tüm eğitim verisi üzerinde son bir kez modelimizi eğiteceğiz.

# 1. En iyi hiperparametreleri seç
best_params <- select_best(svr_tune_results, metric = "rmse")

# 2. İş akışını bu en iyi parametrelerle güncelle
final_svr_workflow <- finalize_workflow(svr_workflow, best_params)

# 3. Final modelini TÜM eğitim verisiyle eğit ve test verisiyle test et
#    last_fit(), bu iki adımı tek bir fonksiyonda birleştirir.
final_svr_fit <- last_fit(final_svr_workflow, data_split)

# 4. Test verisi üzerindeki metrikleri göster
collect_metrics(final_svr_fit)

# Tahminleri görmek isterseniz:
# collect_predictions(final_svr_fit)

# Sıra karar ağacında
# R Projesi: Karar Ağacı ve Hiperparametre Optimizasyonu (tune ile)
# Süreç, SVR için yaptığımızla neredeyse birebir aynı olacak. 
# Sadece model tanımını (parsnip) ve optimize edilecek parametre ızgarasını değiştireceğiz.
# 
# 1. Karar Ağacı Modelini Optimizasyon İçin Tanımlama
# Python'da max_depth, min_samples_split ve min_samples_leaf parametrelerini optimize etmiştik. 
# tidymodels'da bunların karşılıkları tree_depth, min_n ve cost_complexity'dir. 
# cost_complexity (maliyet karmaşıklığı), ağacı budamak (pruning) için kullanılan bir 
# parametredir ve aşırı öğrenmeyi engellemede çok etkilidir.

# 1. Model Tanımlama: Optimize edilecek bir Karar Ağacı
#    'cost_complexity' (budama parametresi), 'tree_depth' (maksimum derinlik)
#    ve 'min_n' (bir düğümdeki minimum örnek sayısı) parametrelerini
#    optimize edeceğimizi tune() ile belirtiyoruz.
dtr_model <- decision_tree(
  cost_complexity = tune(),
  tree_depth = tune(),
  min_n = tune()
) %>%
  set_engine("rpart") %>% # Karar Ağaçları için standart R motoru
  set_mode("regression") 
# İş Akışı ve Optimizasyonu Çalıştırma
# Şimdi bu yeni modeli, daha önce oluşturduğumuz insurance_recipe ile birleştirip optimizasyonu çalıştıracağız.

# İş akışı oluştur
dtr_workflow <- workflow() %>%
  add_recipe(insurance_recipe) %>%
  add_model(dtr_model)

# Optimizasyonu başlat
print("Karar Ağacı için hiperparametre optimizasyonu başlıyor...")
start_time <- Sys.time()

# tune_grid'i çalıştır
# ÖNEMLİ: Karar Ağaçları ölçeklendirmeye ihtiyaç duymaz. Ancak recipe'miz
# ölçeklendirme içeriyor ve bu, ağacın sonucunu değiştirmez.
# Tutarlılık için aynı recipe'i kullanmaya devam ediyoruz.
dtr_tune_results <- tune_grid(
  dtr_workflow,
  resamples = data_folds, # SVR için oluşturduğumuz aynı CV setlerini kullanıyoruz
  grid = 20
)

end_time <- Sys.time()
print(paste("Optimizasyon", round(end_time - start_time, 2), "saniyede tamamlandı."))

# En iyi sonuçları göster
show_best(dtr_tune_results, metric = "rmse")

# Final Modeli Eğitme ve Değerlendirme
# Tıpkı SVR'de olduğu gibi, en iyi parametreleri seçip son modelimizi eğiteceğiz.

# 1. En iyi hiperparametreleri seç
best_params_dtr <- select_best(dtr_tune_results, metric = "rmse")

# 2. İş akışını güncelle
final_dtr_workflow <- finalize_workflow(dtr_workflow, best_params_dtr)

# 3. Final modelini eğit ve test et
final_dtr_fit <- last_fit(final_dtr_workflow, data_split)

# 4. Test verisi üzerindeki metrikleri göster
collect_metrics(final_dtr_fit)

# # KNN regresyon modeline bakalım.
# R Projesi -: KNN Regresyonu ve Hiperparametre Optimizasyonu (tune ile)
# 1. KNN Modelini Optimizasyon İçin Tanımlama
# tidymels'da KNN modeli nearest_neighbor olarak adlandırılır. Optimize edeceğimiz en önemli parametreler:
# 
# neighbors: Komşu sayısı ('K').
# weight_func: Komşuların nasıl ağırlıklandırılacağı ('uniform' veya 'distance''a karşılık gelen fonksiyonlar).
# dist_power: Mesafe metriği (p=1 için Manhattan, p=2 için Euclidean).

# 1. Model Tanımlama: Optimize edilecek bir KNN modeli
#    'neighbors' (K), 'weight_func' ve 'dist_power' (p)
#    parametrelerini optimize edeceğimizi tune() ile belirtiyoruz.
install.packages("kknn")

knn_model <- nearest_neighbor(
  neighbors = tune(),
  weight_func = tune(),
  dist_power = tune()
) %>%
  set_engine("kknn") %>% # KNN için popüler ve hızlı bir motor
  set_mode("regression")

# 2. İş Akışı ve Optimizasyon Izgarası Oluşturma
# Bu sefer, denenecek parametre kombinasyonlarını manuel olarak bir "ızgara" (grid) üzerinde tanımlayalım. 
# Bu, bize süreç üzerinde daha fazla kontrol sağlar.

# İş akışı oluştur
knn_workflow <- workflow() %>%
  add_recipe(insurance_recipe) %>%
  add_model(knn_model)

# Denenecek parametreler için özel bir ızgara oluştur
# expand_grid(), tüm kombinasyonları içeren bir tibble oluşturur.
knn_grid <- expand_grid(
  neighbors = c(3, 5, 7, 9, 11, 15),
  weight_func = c("rectangular", "inv"), # rectangular -> uniform, inv -> distance
  dist_power = c(1, 2) # 1 -> Manhattan, 2 -> Euclidean
)

# Izgaranın ilk birkaç satırına bakalım
# head(knn_grid)
# 3. Optimizasyonu Çalıştırma
# Şimdi tune_grid'i, daha önce oluşturduğumuz çapraz doğrulama setleri ve bu yeni ızgara ile çalıştıralım.

# Optimizasyonu başlat
print("KNN Regresyon için hiperparametre optimizasyonu başlıyor...")
start_time <- Sys.time()

# tune_grid'i çalıştır
knn_tune_results <- tune_grid(
  knn_workflow,
  resamples = data_folds,
  grid = knn_grid # Bu sefer kendi oluşturduğumuz ızgarayı kullanıyoruz
)

end_time <- Sys.time()
print(paste("Optimizasyon", round(end_time - start_time, 2), "saniyede tamamlandı."))

# En iyi sonuçları göster
show_best(knn_tune_results, metric = "rmse")

# 4. Final Modeli Eğitme ve Değerlendirme
# Son adım olarak, en iyi parametrelerle final modelimizi kurup test verisi üzerindeki performansını ölçeceğiz.
# 1. En iyi hiperparametreleri seç
best_params_knn <- select_best(knn_tune_results, metric = "rmse")

# 2. İş akışını güncelle
final_knn_workflow <- finalize_workflow(knn_workflow, best_params_knn)

# 3. Final modelini eğit ve test et
final_knn_fit <- last_fit(final_knn_workflow, data_split)

# 4. Test verisi üzerindeki metrikleri göster
collect_metrics(final_knn_fit)

#---------------- GENEL DEĞERLENDİRME -----------------------------
# R Projesi -: Tüm Modellerin Sonuçlarını Birleştirme ve Nihai Analiz
# Her bir model için last_fit() fonksiyonunu çalıştırdığımızda, 
# sonuçları (final_..._fit değişkenlerinde) zaten elde etmiştik. 
# Şimdi bu sonuçları collect_metrics() ile toplayıp, daha kolay karşılaştırabilmek için 
# tek bir tibble (data frame) içinde birleştireceğiz.
# Her modelin metriklerini topla ve 'model' adında bir sütun ekle
linear_metrics <- final_linear_results %>%
  pivot_wider(names_from = .metric, values_from = .estimate) # Zaten geniş formatta, ama tutarlılık için

svr_metrics <- collect_metrics(final_svr_fit) %>%
  mutate(model = "SVR (Tuned)") %>%
  select(model, .metric, .estimate) %>%
  pivot_wider(names_from = .metric, values_from = .estimate)

dtr_metrics <- collect_metrics(final_dtr_fit) %>%
  mutate(model = "Decision Tree (Tuned)") %>%
  select(model, .metric, .estimate) %>%
  pivot_wider(names_from = .metric, values_from = .estimate)

knn_metrics <- collect_metrics(final_knn_fit) %>%
  mutate(model = "KNN Regressor (Tuned)") %>%
  select(model, .metric, .estimate) %>%
  pivot_wider(names_from = .metric, values_from = .estimate)

# Tüm sonuçları tek bir tibble'da birleştir
# Not: lineer modellerin sonuçları zaten geniş formatta olduğu için yeniden düzenliyoruz
final_r_results <- bind_rows(
  linear_metrics %>% rename(rsq = rsq, rmse = rmse, mae = mae), # Sütun isimlerini eşitle
  svr_metrics %>% select(model, rsq, rmse), # MAE metiği last_fit'te varsayılan değil, şimdilik çıkarıyoruz
  dtr_metrics %>% select(model, rsq, rmse),
  knn_metrics %>% select(model, rsq, rmse)
) %>%
  # En iyi modeli en üste getirmek için RMSE'ye göre sırala
  arrange(rmse)

# Nihai sonuç tablosunu yazdır
print("--- TÜM R MODELLERİNİN NİHAİ PERFORMANS KARŞILAŞTIRMASI ---")
print(final_r_results)

# R Projesi Nihai Sonuç Analizi
# Yukarıdaki tablo, R'da tidymels framework'ü kullanılarak geliştirilen yedi farklı regresyon modelinin test verisi üzerindeki karşılaştırmalı performansını göstermektedir. Modeller, en düşük hata oranından (RMSE) en yükseğe doğru sıralanmıştır.
# 
# Önemli Not: Tablodaki ilk üç modelin (SVR, Decision Tree, KNN) RMSE değerleri logaritmik ölçektedir, çünkü bu modeller logaritmik hedef değişkeni (log_charges) ile eğitilmiştir. Son dört lineer modelin RMSE değeri ise metrik hesaplama yöntemimizdeki farklılıktan dolayı orijinal dolar ölçeğindedir. Bu nedenle, modeller arası karşılaştırmada en adil ve tutarlı metrik R-Kare (rsq) olacaktır.
# 
# 1. Zirvedeki Modeller: SVR ve Karar Ağacı
# 
# Projenin tartışmasız galipleri, sırasıyla %82.9 ve %80.7 R-Kare skorları ile Optimize Edilmiş SVR ve Optimize Edilmiş Karar Ağacı modelleridir.
# Bu sonuç, veri setindeki varyansın %80'inden fazlasını açıklayabildiklerini ve bu problemin çözümü için en uygun model aileleri olduklarını göstermektedir.
# Bu bulgu, Python projesindeki sonuçlarla ve hem Python (lazypredict) hem de R (h2o.automl) ortamlarında yaptığımız ön keşif analizleriyle tamamen tutarlıdır.

# 2. Orta Segment Performans: KNN Regresyonu
# 
# Optimize Edilmiş KNN Regresyonu, %77.3 R-Kare skoru ile oldukça saygın bir performans sergilemiştir.
# Bu sonuç, KNN'yi lineer modellerin belirgin bir şekilde önüne, ancak SVR ve Karar Ağacı'nın bir adım gerisine yerleştirmektedir. Bu sıralama da yine Python projemizle birebir aynıdır.

# 3. Temel Performans Grubu: Lineer Modeller
# 
# Lineer model ailesinin en iyisi, %65.7 R-Kare skoru ile Ridge Regresyon olmuştur. Diğer lineer modeller de bu değere çok yakın sonuçlar vermiştir.
# Doğrusal olmayan modeller ile aralarındaki ~15-17 puanlık R-Kare farkı, lineer varsayımların bu veri seti için ne kadar kısıtlayıcı olduğunun en net kanıtıdır.

