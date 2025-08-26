import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv("insurance.csv")

print(df.head())
print(df.info())
print(df.describe())
print(df.isnull().sum())

# Kategorik değişkenlerin hedef değişkenimiz (charges) üzerindeki etkisini görelim
# Sigara içme durumuna göre masraflar
sns.boxplot(x='smoker', y='charges', data=df)
plt.title('Sigara İçme Durumuna Göre Tıbbi Masraflar')
plt.show()

# Cinsiyete göre masraflar
sns.boxplot(x='sex', y='charges', data=df)
plt.title('Cinsiyete Göre Tıbbi Masraflar')
plt.show()

# Bölgeye göre masraflar
sns.boxplot(x='region', y='charges', data=df)
plt.title('Bölgelere Göre Tıbbi Masraflar')
plt.show()

# Sayısal değişkenlerin ilişkisini inceleyelim
# Yaş ve masraflar arasındaki ilişki
sns.scatterplot(x='age', y='charges', hue='smoker', data=df)
plt.title('Yaş ve Sigara Durumuna Göre Masraflar')
plt.show()

# BMI ve masraflar arasındaki ilişki
sns.scatterplot(x='bmi', y='charges', hue='smoker', data=df)
plt.title('BMI ve Sigara Durumuna Göre Masraflar')
plt.show()


# Sayısal sütunlar arasındaki korelasyonu hesaplayalım
correlation_matrix = df.corr(numeric_only=True)

# Isı haritasını çizelim
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Sayısal Değişkenler Arasındaki Korelasyon Matrisi')
plt.show()

print(correlation_matrix['charges'].sort_values(ascending=False))

# 'charges' sütununun dağılımını inceleyelim
sns.histplot(df['charges'], kde=True, bins=50)
plt.title('Tıbbi Masrafların Dağılımı')
plt.xlabel('Masraflar (Charges)')
plt.ylabel('Frekans')
plt.show()

# Pairplot ile değişkenler arası ilişkilere ve sigara içme durumunun etkisine bakalım
# Bu kod, veri setinin gücünü en iyi gösteren görselleştirmelerden biridir.
sns.pairplot(df, hue='smoker', palette='viridis', corner=True)
plt.show()

# charges sütununun log dönüşümünü alalım ve dağılımını inceleyelim
# Önce orijinal dağılımı ve logaritması alınmış dağılımı karşılaştırmak için
# bir figür ve iki alt grafik oluşturalım.
fig, axes = plt.subplots(1, 2, figsize=(15, 5))

# Sol taraftaki grafik: Orijinal 'charges' dağılımı
sns.histplot(df['charges'], kde=True, ax=axes[0], bins=50)
axes[0].set_title('Dönüşüm Öncesi Orijinal Dağılım')

# 'charges' sütununa log1p dönüşümü uygulayalım
df['log_charges'] = np.log1p(df['charges'])

# Sağ taraftaki grafik: Dönüştürülmüş 'log_charges' dağılımı
sns.histplot(df['log_charges'], kde=True, ax=axes[1], bins=50)
axes[1].set_title('Logaritmik Dönüşüm Sonrası Dağılım')

plt.show()

# data processing encoding one hot encoding ile kategorik değişkenleri sayısala çevirelim
# Kategorik değişkenleri sayısala çevirelim
df_processed = pd.get_dummies(df, columns=['sex', 'smoker', 'region'], drop_first=True)

# drop_first=True, her kategoriden bir sütunu atarak gereksiz tekrarı önler.
# Örneğin, sex_male sütunu 1 ise kadın olmadığını, 0 ise kadın olduğunu anlarız.

print(df_processed.head())

from sklearn.model_selection import train_test_split

# df_processed, One-Hot Encoding sonrası DataFrame'iniz olsun.
# Özellikler (X): 'charges' ve 'log_charges' dışındaki her şey.
X = df_processed.drop(['charges', 'log_charges'], axis=1)

# Hedef (y): Logaritmik dönüşüm uygulanmış 'log_charges' sütunu.
y = df_processed['log_charges']

# Veri setini %80 eğitim, %20 test olarak ayıralım.
# random_state=42 kullanmak, kodu her çalıştırdığınızda aynı ayırma işleminin yapılmasını sağlar,
# bu da sonuçların tekrarlanabilirliği için önemlidir.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Ayırma işleminin doğru yapıldığını kontrol edelim.
print("Eğitim seti özellikleri (X_train) boyutu:", X_train.shape)
print("Test seti özellikleri (X_test) boyutu:", X_test.shape)
print("Eğitim seti hedef (y_train) boyutu:", y_train.shape)
print("Test seti hedef (y_test) boyutu:", y_test.shape)

# Verilerin ölçeklendirilmesi
from sklearn.preprocessing import StandardScaler

# 1. Ölçekleyiciyi oluştur
scaler = StandardScaler()

# 2. Ölçekleyiciyi SADECE eğitim verisi üzerinde eğit ve EĞİTİM VERİSİNİ dönüştür
X_train_scaled = scaler.fit_transform(X_train)

# 3. Aynı (eğitilmiş) ölçekleyiciyi kullanarak TEST VERİSİNİ dönüştür
X_test_scaled = scaler.transform(X_test)

# Dönüşüm sonrası veri NumPy array'ine dönüşür. İsterseniz kontrol için DataFrame'e geri çevirebilirsiniz.
# Bu adım zorunlu değildir, sadece görmek için.
X_train_scaled_df = pd.DataFrame(X_train_scaled, columns=X_train.columns)
print("Ölçeklendirilmiş Eğitim Verisi (İlk 5 Satır):")
print(X_train_scaled_df.head())

# lazy predict ile hızlıca modelleri deneyelim
import lazypredict
from lazypredict.Supervised import LazyRegressor

#---------------------------------------------------------------------
# BU KODLAR ZATEN YUKARIDA VARDI, AKIŞI GÖRMEK İÇİN TEKRAR EDİYORUZ
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import StandardScaler
#
# X = df_processed.drop(['charges', 'log_charges'], axis=1)
# y = df_processed['log_charges']
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
#
# scaler = StandardScaler()
# X_train_scaled = scaler.fit_transform(X_train)
# X_test_scaled = scaler.transform(X_test)
#---------------------------------------------------------------------


# YENİ EKLENECEK KISIM BURASI

print("\n--- LazyPredict ile Genel Model Performans Değerlendirmesi Başlatılıyor ---")

# LazyRegressor'ü oluşturalım.
# verbose=0, her model için detaylı logları kapatır, sadece sonuç tablosunu gösterir.
# ignore_warnings=True, olası kütüphane uyarılarını temizler.
reg = LazyRegressor(verbose=0, ignore_warnings=True, custom_metric=None)

# Modeli ölçeklendirilmiş eğitim ve test verileriyle çalıştıralım
# NOT: LazyPredict, logaritmik ölçekteki 'y' değerlerini kullanır.
# Sonuçları kendi içinde yorumlar.
models, predictions = reg.fit(X_train_scaled, X_test_scaled, y_train, y_test)

# Sonuç tablosunu yazdıralım
print(models)

# Yukarıdaki sonuç tablosunda, farklı regresyon modellerinin performans metriklerini görebilirsiniz.
# R² (R-squared): Modelin açıklayıcılık oranını gösterir.
# RMSE (Root Mean Squared Error): Hata oranını gösterir, ne kadar düşükse o kadar iyidir.
# Time Taken: Modelin eğitilme süresini gösterir.
# Bu metrikler, hangi modelin veri setiniz için en uygun olduğunu belirlemenize yardımcı olur.  

"""
Sonuçlar için yorumlar:
Bu bölümde, projemizin ana modelleme aşamasına geçmeden önce, veri setimizin yapısına en uygun regresyon modellerini belirlemek amacıyla bir ön keşif analizi gerçekleştirilmiştir.
Bu analiz için, onlarca farklı regresyon modelini standart parametrelerle hızlıca deneyip karşılaştıran lazypredict kütüphanesi kullanılmıştır. 
Amaç, hangi model ailelerinin veri setimiz için daha yüksek potansiyel taşıdığını saptamak ve detaylı analiz için seçilecek modelleri bilinçli bir şekilde belirlemektir.

Bulgular ve Analiz

lazypredict çalışması sonucunda elde edilen performans tablosu, modellerin R-Kare (R-Squared) ve Kök Ortalama Kare Hata (RMSE) metriklerine göre sıralanmıştır. 
Tablodan elde edilen temel bulgular şunlardır:

Doğrusal Olmayan Modellerin Üstünlüğü: Tablonun en üst sıralarında GradientBoostingRegressor (R²: 0.87), SVR (R²: 0.86) ve RandomForestRegressor (R²: 0.85) gibi doğrusal olmayan (non-linear) ilişkileri yakalayabilen modellerin yer aldığı görülmektedir. Bu durum, Keşifçi Veri Analizi (EDA) aşamasında tespit ettiğimiz, özellikle 'smoker' (sigara içme durumu) ve 'bmi' (vücut kitle indeksi) gibi değişkenler arasındaki karmaşık etkileşimlerin varlığını doğrulamaktadır.
SVR'nin Güçlü Adaylığı: Ödev kapsamında detaylı incelenecek olan SVR, en iyi performans gösteren ikinci model olarak öne çıkmaktadır. 
Bu, SVR'nin veri setimizdeki yapıları yakalamak için çok güçlü bir aday olduğunu göstermektedir.
Karar Ağacı ve Ağaç Tabanlı Modeller: Tekil DecisionTreeRegressor (R²: 0.81) oldukça iyi bir performans sergilemiştir. 
GradientBoosting, RandomForest gibi daha gelişmiş ağaç tabanlı topluluk modellerinin listenin en başında yer alması, veri setimizin kurallara 
dayalı bölme mantığına (örneğin, "eğer sigara içiyorsa VE bmi > 30 ise...") çok uygun olduğunu kanıtlamaktadır. Bu nedenle, temel yapı taşı olan Karar Ağacı'nı incelemek son derece mantıklıdır.
Lineer Modellerin Performans "Tabanı": LinearRegression, Ridge, LassoCV, Lars ve BayesianRidge gibi tüm lineer modellerin ~0.80 R-Kare skoru civarında kümelendiği görülmektedir. 
Bu, logaritmik dönüşüm sayesinde lineer modellerin dahi oldukça başarılı bir "temel performans (baseline)" seviyesi yakaladığını göstermektedir. 
Bu gruptan LinearRegression'ı temel model olarak seçip, Ridge ve Lasso gibi regülarizasyon tekniklerinin bu temel üzerine bir iyileştirme sağlayıp sağlamadığını incelemek değerli olacaktır.
KNN Regresyon: KNeighborsRegressor da yine ~0.80 R-Kare skoru ile lineer modellerle benzer bir performans grubunda yer almıştır. 
Bu, mesafe bazlı bir algoritmanın da bu problem için geçerli bir seçenek olduğunu göstermektedir.
Sonuç ve Sonraki Adımlar (4. ve 5. Adıma Geçiş)

lazypredict ile yapılan bu ön analiz, model seçimi stratejimizi belirlemede kritik bir rol oynamıştır. 
Elde edilen bulgular ışığında, projenin sonraki adımlarında aşağıdaki modellerin detaylı olarak kodlanmasına, eğitilmesine ve karşılaştırılmasına karar verilmiştir:

Lineer Regresyon Ailesi:
LinearRegression: Problemin temel doğrusal çözümünü ve "baseline" performansını temsil etmesi için.
Ridge, Lasso, ElasticNet: Regülarizasyonun model performansına etkisini ve özellik seçimine yardımcı olup olmadığını gözlemlemek için.
Destek Vektör Regresyonu (SVR): lazypredict tablosunda en üst sıralarda yer alan, doğrusal olmayan ilişkileri yakalama potansiyeli en yüksek modellerden biri olduğu için.
Karar Ağacı Regresyonu (Decision Tree Regressor): Yorumlanabilirliği yüksek, kurallara dayalı yapısıyla veri setimizdeki etkileşimleri 
anlama potansiyeli ve ağaç tabanlı modellerin genel başarısını temsil etmesi için.
K-En Yakın Komşu Regresyonu (KNeighborsRegressor): Farklı bir mantıkla çalışan (örnek bazlı öğrenme) ve 
lineer modellerle benzer bir performans sergileyen bu modeli karşılaştırmaya dahil etmek, model çeşitliliğini artıracaktır.
Bu modeller, hem performans potansiyelleri hem de temel çalışma mantıklarındaki farklılıklar göz önünde bulundurularak seçilmiştir.
 4. ve 5. adımlarda bu modeller tek tek kurulacak, eğitilecek ve performansları orijinal ölçekte (dolar cinsinden) MAE ve R-Kare metrikleri kullanılarak karşılaştırılacaktır.
"""

# Bu yorumlar, projenin ilerleyen adımlarında hangi modellerin neden seçildiğini ve ne tür analizlerin yapılacağını açıklar.
# Ayrıca, bu analiz sürecinin veri bilimi projelerindeki önemini vurgular.
# Bu yorumlar, projenin ilerleyen adımlarında hangi modellerin neden seçildiğini ve ne tür analizlerin yapılacağını açıklar.
# Ayrıca, bu analiz sürecinin veri bilimi projelerindeki önemini vurgular.
# Böylece, modelleme sürecine bilinçli ve veri odaklı bir başlangıç yapılmış olur.

"""
Adım 4.1: Lineer Model Ailesinin Kurulması ve Değerlendirilmesi
Bu bölümde, projemizin temelini oluşturacak olan lineer modelleri analiz edeceğiz. 
Tüm modeller, daha önce hazırlanan ölçeklendirilmiş eğitim verisi (X_train_scaled, y_train) üzerinde eğitilecek ve performansları, 
logaritmik tahminlerin orijinal dolar birimine geri çevrilmesiyle test verisi üzerinde ölçülecektir.
"""

# Gerekli kütüphaneleri ve metrikleri import edelim
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import pandas as pd
import numpy as np

# Sonuçları saklamak için bir liste oluşturalım
linear_models_results = []

# y_test'i orijinal ölçeğine çevirelim
y_test_original = np.expm1(y_test)

# --- 1. Basit Lineer Regresyon (Baseline Model) ---
print("1. Lineer Regresyon Modeli Eğitiliyor...")
lr = LinearRegression()
lr.fit(X_train_scaled, y_train)

# Tahminleri yap ve geri çevir
y_pred_log_lr = lr.predict(X_test_scaled)
y_pred_lr = np.expm1(y_pred_log_lr)

# Metrikleri hesapla
r2_lr = r2_score(y_test_original, y_pred_lr)
mae_lr = mean_absolute_error(y_test_original, y_pred_lr)
mse_lr = mean_squared_error(y_test_original, y_pred_lr)
rmse_lr = np.sqrt(mse_lr)
linear_models_results.append({'Model': 'Linear Regression', 'R2 Score': r2_lr, 'MAE': mae_lr, 'RMSE': rmse_lr})
print(f"   - Tamamlandı. R2 Score: {r2_lr:.4f}, MAE: {mae_lr:.2f}, RMSE: {rmse_lr:.2f}\n")


# --- 2. Ridge Regresyon (L2 Regularization) ---
print("2. Ridge Regresyon Modeli Eğitiliyor...")
ridge = Ridge(alpha=0.01,random_state=42)
ridge.fit(X_train_scaled, y_train)

# Tahminleri yap ve geri çevir
y_pred_log_ridge = ridge.predict(X_test_scaled)
y_pred_ridge = np.expm1(y_pred_log_ridge)

# Metrikleri hesapla
r2_ridge = r2_score(y_test_original, y_pred_ridge)
mae_ridge = mean_absolute_error(y_test_original, y_pred_ridge)
mse_ridge = mean_squared_error(y_test_original, y_pred_ridge)
rmse_ridge = np.sqrt(mse_ridge)
linear_models_results.append({'Model': 'Ridge Regression', 'R2 Score': r2_ridge, 'MAE': mae_ridge, 'RMSE': rmse_ridge})
print(f"   - Tamamlandı. R2 Score: {r2_ridge:.4f}, MAE: {mae_ridge:.2f}, RMSE: {rmse_ridge:.2f}\n")


# --- 3. Lasso Regresyon (L1 Regularization) ---
print("3. Lasso Regresyon Modeli Eğitiliyor...")
lasso = Lasso(alpha=0.01, random_state=42) # alpha değeri regülarizasyon gücünü belirler
lasso.fit(X_train_scaled, y_train)

# Tahminleri yap ve geri çevir
y_pred_log_lasso = lasso.predict(X_test_scaled)
y_pred_lasso = np.expm1(y_pred_log_lasso)

# Metrikleri hesapla
r2_lasso = r2_score(y_test_original, y_pred_lasso)
mae_lasso = mean_absolute_error(y_test_original, y_pred_lasso)
mse_lasso = mean_squared_error(y_test_original, y_pred_lasso)
rmse_lasso = np.sqrt(mse_lasso)
linear_models_results.append({'Model': 'Lasso Regression', 'R2 Score': r2_lasso, 'MAE': mae_lasso, 'RMSE': rmse_lasso})
print(f"   - Tamamlandı. R2 Score: {r2_lasso:.4f}, MAE: {mae_lasso:.2f}, RMSE: {rmse_lasso:.2f}\n")


# --- 4. Elastic Net Regresyon (L1 & L2 Combination) ---
print("4. Elastic Net Modeli Eğitiliyor...")
elastic_net = ElasticNet(alpha=0.001, random_state=42) # alpha değeri regülarizasyon gücünü belirler
elastic_net.fit(X_train_scaled, y_train)

# Tahminleri yap ve geri çevir
y_pred_log_elastic = elastic_net.predict(X_test_scaled)
y_pred_elastic = np.expm1(y_pred_log_elastic)

# Metrikleri hesapla
r2_elastic = r2_score(y_test_original, y_pred_elastic)
mae_elastic = mean_absolute_error(y_test_original, y_pred_elastic)
mse_elastic = mean_squared_error(y_test_original, y_pred_elastic)
rmse_elastic = np.sqrt(mse_elastic)
linear_models_results.append({'Model': 'Elastic Net', 'R2 Score': r2_elastic, 'MAE': mae_elastic, 'RMSE': rmse_elastic})
print(f"   - Tamamlandı. R2 Score: {r2_elastic:.4f}, MAE: {mae_elastic:.2f}, RMSE: {rmse_elastic:.2f}\n")


# --- Sonuçları Toplu Halde Gösterelim ---
results_df = pd.DataFrame(linear_models_results)
print("--- LİNEER MODEL AİLESİ PERFORMANS ÖZETİ (RMSE DAHİL) ---")
print(results_df)
# Sonuçları RMSE'ye göre sıralayalım
results_df_sorted = results_df.sort_values(by='RMSE')
print("\n--- RMSE'ye Göre Sıralanmış Sonuçlar ---")
print(results_df_sorted)
"""
Sonuçların Analizi ve Yorumlanması

Lasso'nun Başarısı (En Önemli Bulgu): Sıralanmış listeye baktığımızda, en iyi performansı (en düşük RMSE ve MAE, en yüksek R-Kare) 
Lasso Regresyon modelinin sergilediğini görüyoruz. Bu tesadüf değil.
Anlamı: Lasso Regresyon (L1 regülarizasyonu), önemsiz veya daha az önemli gördüğü özelliklerin katsayılarını tam olarak sıfır yapma yeteneğine sahiptir. 
Bu, bir nevi otomatik "özellik seçimi" (feature selection) yapması demektir. Sonuçlarınız, veri setinizdeki bazı özelliklerin elenmesinin, 
gürültüyü azaltarak daha iyi bir tahmin modeline yol açtığını gösteriyor.
Ridge ve Lineer Regresyonun Yakınlığı: Ridge ve Linear Regression modellerinin performansları neredeyse aynı.
Anlamı: Ridge (L2 regülarizasyonu), katsayıları küçültür ama asla tam olarak sıfır yapmaz; tüm özellikleri modelde tutar. 
Performansının temel lineer regresyondan belirgin şekilde daha iyi olmaması, modelde çok ciddi bir "multicollinearity" (özelliklerin birbiriyle yüksek korelasyonu) 
sorunu olmadığını veya L2 regülarizasyonunun bu veri setinde büyük bir fark yaratmadığını düşündürür.
Genel İyileşme: Tüm lineer modeller artık mantıklı ve tutarlı sonuçlar veriyor. R-Kare değerlerinin ~0.61-0.63 aralığında olması, 
bu model ailesinin, tıbbi masraflardaki varyansın yaklaşık %61-63'ünü açıklayabildiğini gösteriyor. RMSE değerinin ~7600-7800$ aralığında olması ise, bu modellerin tahminlerinin ortalama olarak bu civarda bir hata payına sahip olduğunu belirtiyor.
Ödev Raporu İçin Çıkarımlar

"Lineer model ailesi, hiperparametre ayarlaması sonrasında tutarlı sonuçlar vermiştir. 
Varsayılan alpha değerleriyle başarısız olan Lasso ve ElasticNet, alpha değerleri düşürüldüğünde rekabetçi hale gelmiştir."
"Bu aile içindeki en başarılı model, 7617$ RMSE değeri ile Lasso Regresyon olmuştur. 
Bu durum, Lasso'nun otomatik özellik seçimi yaparak gürültüyü azaltmasının ve daha genelleştirilebilir bir model oluşturmasının bir sonucu olarak yorumlanabilir."
"Lineer modellerin ulaştığı en iyi R-Kare skoru ~0.63'tür. Bu, projemiz için bir temel performans (baseline) seviyesi olarak kabul edilecektir. 
Sonraki adımlarda incelenecek olan SVR ve Karar Ağacı gibi doğrusal olmayan modellerin bu temel performans seviyesini aşıp aşamayacağı test edilecektir."
"""

# SVR ile devam edelim
from sklearn.svm import SVR # Support Vector Regressor
from sklearn.model_selection import GridSearchCV    
import time

if __name__ == '__main__':

    # --- SVR Modeli ve Hiperparametre Optimizasyonu ---
    print("4. SVR Modeli için Hiperparametre Optimizasyonu Başlatılıyor...")
    start_time = time.time()

    # 1. Optimize edilecek parametreler için bir 'ızgara' (grid) tanımlayalım
    # Bu değerler, denenecek olan C ve gamma kombinasyonlarını içerir.
    param_grid = {
        'C': [0.1, 1, 10, 100],
        'gamma': ['scale', 'auto', 0.1, 0.01],
        'kernel': ['rbf'] # En iyi aday olduğu için sadece 'rbf' kernelini deniyoruz.
    }

    # 2. GridSearchCV nesnesini oluşturalım
    #   - estimator: Optimize edilecek model (SVR)
    #   - param_grid: Denenecek parametreler
    #   - cv=5: 5-katlı çapraz doğrulama (daha güvenilir sonuçlar için)
    #   - scoring: En iyi parametreyi seçerken kullanılacak metrik. 'neg_mean_squared_error' MSE'yi minimize etmeye çalışır.
    #   - n_jobs=-1: Bilgisayarın tüm işlemci çekirdeklerini kullanarak aramayı hızlandırır.
    #   - verbose=2: Arama sırasında ilerleme durumunu gösterir.
    grid_search = GridSearchCV(
        estimator=SVR(),
        param_grid=param_grid,
        cv=5,
        scoring='neg_mean_squared_error',
        n_jobs=-1,
        verbose=2
    )

    # 3. Grid Search'ü ölçeklendirilmiş eğitim verisiyle başlatalım
    # Bu adım, tüm kombinasyonları denediği için biraz zaman alabilir.
    grid_search.fit(X_train_scaled, y_train)

    end_time = time.time()
    print(f"\nGrid Search {end_time - start_time:.2f} saniyede tamamlandı.")

    # 4. En iyi parametreleri ve en iyi skoru yazdıralım
    print("\nEn iyi parametreler bulundu:")
    print(grid_search.best_params_)

    # 5. En iyi modeli kullanarak tahmin yapalım
    # GridSearchCV, en iyi parametrelerle modeli zaten yeniden eğitmiştir.
    # Bu eğilmiş modele 'best_estimator_' ile erişebiliriz.
    best_svr = grid_search.best_estimator_

    # Tahminleri logaritmik ölçekte yap ve orijinal ölçeğe geri çevir
    y_pred_log_svr = best_svr.predict(X_test_scaled)
    y_pred_svr = np.expm1(y_pred_log_svr)

    # 6. Metrikleri hesaplayalım
    r2_svr = r2_score(y_test_original, y_pred_svr)
    mae_svr = mean_absolute_error(y_test_original, y_pred_svr)
    mse_svr = mean_squared_error(y_test_original, y_pred_svr)
    rmse_svr = np.sqrt(mse_svr)


    # Sonuçları saklamak için bir sözlük oluşturalım
    svr_result = {
        'Model': 'SVR (Tuned)',
        'R2 Score': r2_svr,
        'MAE': mae_svr,
        'RMSE': rmse_svr,
        'Best Params': [grid_search.best_params_] # Parametreleri de saklayalım
    }

    # SVR sonucunu gösterelim
    print("\n--- Optimize Edilmiş SVR Modeli Performansı ---")
    print(f"R2 Score: {r2_svr:.4f}")
    print(f"MAE: {mae_svr:.2f}")
    print(f"RMSE: {rmse_svr:.2f}")

    # Bu sonucu, daha sonraki genel karşılaştırma için ana sonuç listemize ekleyebiliriz.
    # Örneğin: all_models_results.append(svr_result)

    # Şimdi sıra karar ağacı modelinde
    from sklearn.tree import DecisionTreeRegressor
    # --- Karar Ağacı Modeli ve Hiperparametre Optimizasyonu ---
    print("5. Karar Ağacı Modeli için Hiperparametre Optimizasyonu Başlatılıyor...")
    start_time = time.time()

    # 1. Optimize edilecek parametreler için bir 'ızgara' (grid) tanımlayalım
    param_grid = {
        'criterion': ['squared_error', 'absolute_error'],
        'max_depth': [3, 5, 7, 10, None], # None, derinlik sınırı olmadığını belirtir
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }

    # 2. GridSearchCV nesnesini oluşturalım
    # Ayarlar SVR ile benzer, sadece estimator değişiyor.
    grid_search_dtr = GridSearchCV(
        estimator=DecisionTreeRegressor(random_state=42),
        param_grid=param_grid,
        cv=5,
        scoring='neg_mean_squared_error',
        n_jobs=-1,
        verbose=2
    )

    # 3. Grid Search'ü ölçeklendirilmiş eğitim verisiyle başlatalım
    # NOT: Karar Ağaçları ölçeklendirmeye ihtiyaç duymaz, ancak tutarlılık
    # açısından ve tüm modelleri aynı veri üzerinde denemek için ölçeklendirilmiş
    # veriyi kullanmaya devam ediyoruz. Performansı etkilemeyecektir.
    grid_search_dtr.fit(X_train_scaled, y_train)

    end_time = time.time()
    print(f"\nGrid Search {end_time - start_time:.2f} saniyede tamamlandı.")

    # 4. En iyi parametreleri ve en iyi skoru yazdıralım
    print("\nEn iyi parametreler bulundu:")
    print(grid_search_dtr.best_params_)

    # 5. En iyi modeli kullanarak tahmin yapalım
    best_dtr = grid_search_dtr.best_estimator_

    # Tahminleri logaritmik ölçekte yap ve orijinal ölçeğe geri çevir
    y_pred_log_dtr = best_dtr.predict(X_test_scaled)
    y_pred_dtr = np.expm1(y_pred_log_dtr)

    # 6. Metrikleri hesaplayalım
    r2_dtr = r2_score(y_test_original, y_pred_dtr)
    mae_dtr = mean_absolute_error(y_test_original, y_pred_dtr)
    mse_dtr = mean_squared_error(y_test_original, y_pred_dtr)
    rmse_dtr = np.sqrt(mse_dtr)

    # Sonuçları saklamak için bir sözlük oluşturalım
    dtr_result = {
        'Model': 'Decision Tree (Tuned)',
        'R2 Score': r2_dtr,
        'MAE': mae_dtr,
        'RMSE': rmse_dtr,
        'Best Params': [grid_search_dtr.best_params_]
    }

    # Karar Ağacı sonucunu gösterelim
    print("\n--- Optimize Edilmiş Karar Ağacı Modeli Performansı ---")
    print(f"R2 Score: {r2_dtr:.4f}")
    print(f"MAE: {mae_dtr:.2f}")
    print(f"RMSE: {rmse_dtr:.2f}")

    # Bu sonucu da ana sonuç listemize ekleyebiliriz.
    # Örneğin: all_models_results.append(dtr_result)

    # Son olarak KNN modelini de ekleyelim
    from sklearn.neighbors import KNeighborsRegressor
    # --- KNN Modeli ve Hiperparametre Optimizasyonu ---
    print("6. KNN Regresyon Modeli için Hiperparametre Optimizasyonu Başlatılıyor...")
    start_time = time.time()

    # 1. Optimize edilecek parametreler için bir 'ızgara' (grid) tanımlayalım
    param_grid = {
        'n_neighbors': [3, 5, 7, 9, 11, 15],
        'weights': ['uniform', 'distance'],
        'p': [1, 2] # 1: Manhattan, 2: Euclidean
    }

    # 2. GridSearchCV nesnesini oluşturalım
    grid_search_knn = GridSearchCV(
        estimator=KNeighborsRegressor(),
        param_grid=param_grid,
        cv=5,
        scoring='neg_mean_squared_error',
        n_jobs=-1,
        verbose=2
    )

    # 3. Grid Search'ü ölçeklendirilmiş eğitim verisiyle başlatalım
    # KNN'in mesafeye dayalı doğası gereği ölçeklendirilmiş veri kullanmak ZORUNLUDUR.
    grid_search_knn.fit(X_train_scaled, y_train)

    end_time = time.time()
    print(f"\nGrid Search {end_time - start_time:.2f} saniyede tamamlandı.")

    # 4. En iyi parametreleri ve en iyi skoru yazdıralım
    print("\nEn iyi parametreler bulundu:")
    print(grid_search_knn.best_params_)

    # 5. En iyi modeli kullanarak tahmin yapalım
    best_knn = grid_search_knn.best_estimator_

    # Tahminleri logaritmik ölçekte yap ve orijinal ölçeğe geri çevir
    y_pred_log_knn = best_knn.predict(X_test_scaled)
    y_pred_knn = np.expm1(y_pred_log_knn)

    # 6. Metrikleri hesaplayalım
    r2_knn = r2_score(y_test_original, y_pred_knn)
    mae_knn = mean_absolute_error(y_test_original, y_pred_knn)
    mse_knn = mean_squared_error(y_test_original, y_pred_knn)
    rmse_knn = np.sqrt(mse_knn)

    # Sonuçları saklamak için bir sözlük oluşturalım
    knn_result = {
        'Model': 'KNN Regressor (Tuned)',
        'R2 Score': r2_knn,
        'MAE': mae_knn,
        'RMSE': rmse_knn,
        'Best Params': [grid_search_knn.best_params_]
    }

    # KNN sonucunu gösterelim
    print("\n--- Optimize Edilmiş KNN Regresyon Modeli Performansı ---")
    print(f"R2 Score: {r2_knn:.4f}")
    print(f"MAE: {mae_knn:.2f}")
    print(f"RMSE: {rmse_knn:.2f}")

    # Bu sonucu da ana sonuç listemize ekleyebiliriz.
    # Örneğin: all_models_results.append(knn_result)

# Tüm Modellerin Sonuçlarını Birleştirme ve Nihai Analiz

################################################################################
# ADIM 1: TÜM SONUÇLARI TUTACAK ANA LİSTEYİ OLUŞTUR
################################################################################
all_models_results = []

# y_test'i orijinal ölçeğine çevirelim (Tüm modeller için ortak)
y_test_original = np.expm1(y_test)

################################################################################
# BÖLÜM 4.1: Lineer Model Ailesi
################################################################################
print("--- Lineer Model Ailesi Çalıştırılıyor ---")
# --- Lineer Regresyon ---
lr = LinearRegression().fit(X_train_scaled, y_train)
y_pred_lr = np.expm1(lr.predict(X_test_scaled))
all_models_results.append({
    'Model': 'Linear Regression',
    'R2 Score': r2_score(y_test_original, y_pred_lr),
    'MAE': mean_absolute_error(y_test_original, y_pred_lr),
    'RMSE': np.sqrt(mean_squared_error(y_test_original, y_pred_lr))
})

# --- Ridge Regresyon ---
ridge = Ridge(alpha=0.01, random_state=42).fit(X_train_scaled, y_train)
y_pred_ridge = np.expm1(ridge.predict(X_test_scaled))
all_models_results.append({
    'Model': 'Ridge Regression',
    'R2 Score': r2_score(y_test_original, y_pred_ridge),
    'MAE': mean_absolute_error(y_test_original, y_pred_ridge),
    'RMSE': np.sqrt(mean_squared_error(y_test_original, y_pred_ridge))
})

# --- Lasso Regresyon ---
lasso = Lasso(alpha=0.01, random_state=42).fit(X_train_scaled, y_train)
y_pred_lasso = np.expm1(lasso.predict(X_test_scaled))
all_models_results.append({
    'Model': 'Lasso Regression',
    'R2 Score': r2_score(y_test_original, y_pred_lasso),
    'MAE': mean_absolute_error(y_test_original, y_pred_lasso),
    'RMSE': np.sqrt(mean_squared_error(y_test_original, y_pred_lasso))
})
print("Lineer modeller tamamlandı.\n")


################################################################################
# BÖLÜM 4.2: SVR (Hiperparametre Optimizasyonu ile)
################################################################################
print("--- SVR Modeli Çalıştırılıyor (Grid Search) ---")
param_grid_svr = {'C': [1, 10, 100], 'gamma': ['scale', 0.1, 0.01], 'kernel': ['rbf']}
grid_search_svr = GridSearchCV(SVR(), param_grid_svr, cv=5, scoring='neg_mean_squared_error', n_jobs=-1, verbose=0)
grid_search_svr.fit(X_train_scaled, y_train)
best_svr = grid_search_svr.best_estimator_
y_pred_svr = np.expm1(best_svr.predict(X_test_scaled))
all_models_results.append({
    'Model': 'SVR (Tuned)',
    'R2 Score': r2_score(y_test_original, y_pred_svr),
    'MAE': mean_absolute_error(y_test_original, y_pred_svr),
    'RMSE': np.sqrt(mean_squared_error(y_test_original, y_pred_svr))
})
print(f"SVR tamamlandı. En iyi parametreler: {grid_search_svr.best_params_}\n")


################################################################################
# BÖLÜM 4.3: Karar Ağacı (Hiperparametre Optimizasyonu ile)
################################################################################
print("--- Karar Ağacı Modeli Çalıştırılıyor (Grid Search) ---")
param_grid_dtr = {'max_depth': [3, 5, 7, 10], 'min_samples_split': [2, 5, 10], 'min_samples_leaf': [1, 2, 4]}
grid_search_dtr = GridSearchCV(DecisionTreeRegressor(random_state=42), param_grid_dtr, cv=5, scoring='neg_mean_squared_error', n_jobs=-1, verbose=0)
grid_search_dtr.fit(X_train_scaled, y_train)
best_dtr = grid_search_dtr.best_estimator_
y_pred_dtr = np.expm1(best_dtr.predict(X_test_scaled))
all_models_results.append({
    'Model': 'Decision Tree (Tuned)',
    'R2 Score': r2_score(y_test_original, y_pred_dtr),
    'MAE': mean_absolute_error(y_test_original, y_pred_dtr),
    'RMSE': np.sqrt(mean_squared_error(y_test_original, y_pred_dtr))
})
print(f"Karar Ağacı tamamlandı. En iyi parametreler: {grid_search_dtr.best_params_}\n")


################################################################################
# BÖLÜM 4.4: KNN Regresyon (Hiperparametre Optimizasyonu ile)
################################################################################
print("--- KNN Regresyon Modeli Çalıştırılıyor (Grid Search) ---")
param_grid_knn = {'n_neighbors': [3, 5, 7, 9, 11], 'weights': ['uniform', 'distance'], 'p': [1, 2]}
grid_search_knn = GridSearchCV(KNeighborsRegressor(), param_grid_knn, cv=5, scoring='neg_mean_squared_error', n_jobs=-1, verbose=0)
grid_search_knn.fit(X_train_scaled, y_train)
best_knn = grid_search_knn.best_estimator_
y_pred_knn = np.expm1(best_knn.predict(X_test_scaled))
all_models_results.append({
    'Model': 'KNN Regressor (Tuned)',
    'R2 Score': r2_score(y_test_original, y_pred_knn),
    'MAE': mean_absolute_error(y_test_original, y_pred_knn),
    'RMSE': np.sqrt(mean_squared_error(y_test_original, y_pred_knn))
})
print(f"KNN tamamlandı. En iyi parametreler: {grid_search_knn.best_params_}\n")


################################################################################
# ADIM 3: NİHAİ SONUÇ TABLOSUNU OLUŞTUR
################################################################################
final_results_df = pd.DataFrame(all_models_results)

# En iyi modeli en üste getirmek için sonuçları RMSE'ye göre sıralayalım
final_results_df_sorted = final_results_df.sort_values(by='RMSE')

print("--- TÜM MODELLERİN NİHAİ PERFORMANS KARŞILAŞTIRMASI ---")
print(final_results_df_sorted)

"""
Nihai Model Performans Analizi
Proje kapsamında geliştirilen ve optimize edilen regresyon modellerinin karşılaştırmalı performans sonuçları yukarıdaki tabloda özetlenmiştir. 
Modeller, en düşük hata oranını (RMSE) gösterenden en yükseğe doğru sıralanmıştır. Tabloyu analiz ettiğimizde şu önemli sonuçlara ulaşmaktayız:

1. En Başarılı Modeller: Karar Ağacı ve SVR

Zirvedeki Yakın Rekabet: En yüksek performansı, birbirine çok yakın sonuçlarla Optimize Edilmiş Karar Ağacı ve Optimize Edilmiş SVR modelleri sergilemiştir. 
Her iki model de R-Kare (R²) skorunda %87'lik etkileyici bir başarıya ulaşarak veri setindeki değişkenliğin çok büyük bir kısmını açıklamayı başarmıştır.
Hata Oranlarının Karşılaştırması:
Karar Ağacı, 4452$ RMSE ile en düşük kök ortalama kare hataya sahiptir. Bu, modelin büyük hataları cezalandırma eğilimini yansıtan metrikte en başarılı olduğunu gösterir.
SVR ise 2086$ MAE (Ortalama Mutlak Hata) ile bu metrikte en başarılı modeldir. Bu da SVR'nin tahminlerinin ortalama olarak gerçek değerden en az sapan model olduğunu gösterir.
Yorum: Bu iki model arasındaki küçük farklar, hangi hata türüne daha fazla önem verildiğine göre değişebilir. 
Ancak genel olarak, her ikisinin de bu problem için en uygun modeller olduğu açıktır. lazypredict ile yaptığımız ön analizde bu iki model ailesinin potansiyeli öngörülmüştü 
ve detaylı analiz bu öngörüyü doğrulamıştır.

2. KNN Regresyon'un Orta Seviye Başarısı

Net Bir Ayrım: Optimize edilmiş KNN Regresyonu, %80 R-Kare skoru ve ~5567$ RMSE ile iyi bir performans göstermiş, ancak zirvedeki iki modelin belirgin şekilde gerisinde kalmıştır.
Yorum: KNN'nin bu "orta segment" performansı, mesafe bazlı basit mantığının, SVR ve Karar Ağacı'nın karmaşık ilişkileri modelleme yeteneği kadar güçlü olmadığını göstermektedir. 
Yine de lineer modellerden daha başarılıdır.

3. Lineer Modellerin Sınırları

Temel Performans Seviyesi: Lineer Regresyon, Ridge ve Lasso modelleri, %61-63 R-Kare ve ~7600-7800$ RMSE aralığında, birbirine çok yakın sonuçlarla tablonun en alt sıralarında yer almıştır.
Yorum: Bu sonuçlar, lineer modellerin bu problem için yetersiz kaldığını açıkça ortaya koymaktadır.
Daha önce EDA aşamasında tespit ettiğimiz, değişkenler arasındaki doğrusal olmayan ve karmaşık etkileşimler (örneğin sigara içme durumunun BMI etkisini katlaması gibi), 
bu modellerin varsayımlarıyla çelişmektedir. Bu nedenle, ne kadar optimize edilirse edilsinler, performanslarının belirli bir tavanı aşamaması beklenen bir sonuçtur. 
Lasso Regresyon'un bu aile içinde en iyi olması, özellik seçimi yaparak gürültüyü bir miktar azaltmasının bir sonucudur.
Projenin Genel Sonucu ve Nihai Karar
Bu proje, tıbbi masrafların tahmini gibi karmaşık bir problemde, doğru makine öğrenmesi modelini seçmenin ve optimize etmenin ne kadar kritik olduğunu göstermiştir.

Nihai Karar: Eğer bu model bir şirkette kullanılacak olsaydı, Optimize Edilmiş Karar Ağacı veya Optimize Edilmiş SVR modellerinden biri tercih edilirdi. 
Karar Ağacı'nın sonuçlarının (örneğin, "Eğer sigara içiyorsa VE yaşı 45'ten büyükse...") insanlar tarafından daha kolay yorumlanabilir olması, 
iş birimlerine açıklama yapma kolaylığı açısından ona küçük bir avantaj sağlayabilir.
Ancak, SVR'nin MAE'deki üstün performansı, özellikle ortalama hatanın minimize edilmesinin kritik olduğu durumlarda onu cazip kılar.
Sonuç olarak, bu proje, veri bilimi projelerinde model seçimi ve hiperparametre optimizasyonunun önemini vurgulayan kapsamlı bir vaka çalışması olmuştur. 
Elde edilen sonuçlar, gelecekte benzer problemlerle karşılaşıldığında yol gösterici olacaktır"""