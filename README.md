# Asthma Risk Prediction & Epidemiological ML Analysis

Bu çalışma; astım hastalığı veri seti üzerinde epidemiyolojik risk faktörlerini istatistiksel modellerle inceleyen, sınıf dengesizliğini (class imbalance) ele alan ve çoklu makine öğrenmesi algoritmalarını karşılaştıran uçtan uca bir veri bilimi projesidir.

---

## 📌 Proje Kapsamı ve Yöntem

### 1. Veri Ön İşleme & Epidemiyolojik Analiz
* Yaş ve cinsiyet kırılımlarında gruplandırma (`Child` vs. `Adult`).
* Demografik gruplara göre ham prevalans (prevalence) oranlarının tespiti ve görselleştirilmesi.

### 2. Biyoistatistiksel Risk Analizi (Statsmodels)
* Çok değişkenli **Lojistik Regresyon** modeli kurularak çevresel faktörler (hava kirliliği, polen, toz maruziyeti), genetik geçmiş ve komorbiditeler incelenmiştir.
* Her bir risk faktörü için **Odds Ratio (OR)**, %95 Güven Aralığı (CI) ve p-değerleri hesaplanmıştır.

### 3. Makine Öğrenmesi Benchmark & Karşılaştırma
Standart ve sınıf ağırlıklı (`class_weight="balanced"`) olmak üzere iki farklı senaryoda 5 farklı sınıflandırma algoritması eğitilmiştir:
* **Logistic Regression**
* **Decision Tree**
* **k-Nearest Neighbors (kNN)** *(StandardScaler ile ölçeklenmiş)*
* **Random Forest**
* **Support Vector Machine (SVM)** *(StandardScaler ile ölçeklenmiş)*

### 4. Medikal Performans Metrikleri
Sağlık verilerinde hayati önem taşıyan metrikler Confusion Matrix üzerinden elde edilerek modeller kıyaslanmıştır:
* **Accuracy** (Doğruluk)
* **Sensitivity / Recall** (Duyarlılık – Gerçek pozitifleri yakalama oranı)
* **Specificity** (Özgüllük – Gerçek negatifleri ayırt etme oranı)
* **Balanced Accuracy** (Dengeli Doğruluk)

---

## 🛠️ Kullanılan Teknolojiler

* **Dil:** Python
* **Veri Analitiği:** Pandas, NumPy
* **İstatistik & Modelleme:** Statsmodels
* **Makine Öğrenmesi:** Scikit-Learn
* **Görselleştirme:** Matplotlib

---

## 🚀 Projeyi Çalıştırma
Gerekli kütüphaneleri yükleyin:
```bash
pip install pandas numpy statsmodels scikit-learn matplotlib
Gerekli kütüphaneleri yükleyin:
```bash
pip install pandas numpy statsmodels scikit-learn matplotlib
