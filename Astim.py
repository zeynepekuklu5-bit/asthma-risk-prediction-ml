import pandas as pd
import numpy as np

import statsmodels.api as sm
from statsmodels.formula.api import logit
pd.set_option("display.max_columns", None)
pd.set_option("display.width", 200)

# Makine öğrenmesi
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

# 1. Veri Setini Yükle

df = pd.read_csv("asthma_disease_data.csv")
print("\n--- Veri Seti Yüklendi ---")
print("Gözlem x Değişken:", df.shape)

# 2. Veri Ön İşleme

# Yaş grubu
df["Age_Group"] = df["Age"].apply(lambda x: "Child" if x < 18 else "Adult")
df["Age_Group"] = pd.Categorical(df["Age_Group"], categories=["Child", "Adult"])

# Cinsiyet
df["Gender_Cat"] = df["Gender"].replace({0: "Female", 1: "Male"})
df["Gender_Cat"] = pd.Categorical(df["Gender_Cat"], categories=["Male", "Female"])

# 3. Ham Prevalans (TABLO 1)

prevalence_table = (
    df.groupby(["Age_Group", "Gender_Cat"], observed=False)["Diagnosis"]
    .agg(Total_N="count", Asthma_N="sum")
)

prevalence_table["Prevalence (%)"] = (
    prevalence_table["Asthma_N"] / prevalence_table["Total_N"] * 100
).round(2)

print("\n--- TABLO 1: Ham Prevalans ---")
print(prevalence_table)

# ======================================================
# 4. Lojistik Regresyon (TABLO 2)
# ======================================================

logit_model = logit(
    "Diagnosis ~ C(Age_Group, Treatment('Adult')) * "
    "C(Gender_Cat, Treatment('Female')) + "
    "BMI + C(Smoking) + C(FamilyHistoryAsthma) + "
    "C(HistoryOfAllergies) + C(Eczema) + C(HayFever) + "
    "C(GastroesophagealReflux) + "
    "PollutionExposure + PollenExposure + DustExposure",
    data=df
).fit(disp=False)

or_table = pd.DataFrame({
    "Odds Ratio (OR)": np.exp(logit_model.params),
    "CI 2.5%": np.exp(logit_model.conf_int()[0]),
    "CI 97.5%": np.exp(logit_model.conf_int()[1]),
    "p-value": logit_model.pvalues
})

or_table = or_table.iloc[1:].round(3)

print("\n--- TABLO 2: Lojistik Regresyon (OR) ---")
print(or_table)


# ======================================================
# 5.A Makine Öğrenmesi Analizi
# ======================================================

y = df["Diagnosis"]

X = df[
    [
        "Age_Group", "Gender_Cat", "BMI", "Smoking",
        "FamilyHistoryAsthma", "HistoryOfAllergies",
        "Eczema", "HayFever", "GastroesophagealReflux",
        "PollutionExposure", "PollenExposure", "DustExposure"
    ]
]

X = pd.get_dummies(X, drop_first=True)

# Eğitim / Test bölünmesi
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42
)

# Ölçekleme (kNN ve SVM için)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

models = {
    "Logistic Regression (ML)": LogisticRegression(max_iter=1000),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "kNN": KNeighborsClassifier(n_neighbors=5),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "SVM": SVC()
}

performance_results = []

for name, model in models.items():

    if name in ["kNN", "SVM"]:
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
    else:
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

    # Confusion matrix
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

    accuracy = (tp + tn) / (tp + tn + fp + fn)
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else np.nan
    specificity = tn / (tn + fp) if (tn + fp) > 0 else np.nan

    performance_results.append([
        name,
        round(accuracy, 3),
        round(sensitivity, 3),
        round(specificity, 3)
    ])

# ======================================================
# 5B. Makine Öğrenmesi – Dengeli (class_weight="balanced")
# ======================================================

balanced_models = {
    "Logistic Regression (Balanced)": LogisticRegression(
        max_iter=1000, class_weight="balanced"
    ),
    "Random Forest (Balanced)": RandomForestClassifier(
        n_estimators=100, random_state=42, class_weight="balanced"
    ),
    "SVM (Balanced)": SVC(class_weight="balanced")
}

balanced_results = []

for name, model in balanced_models.items():

    if "SVM" in name:
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
    else:
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

    accuracy = (tp + tn) / (tp + tn + fp + fn)
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else np.nan
    specificity = tn / (tn + fp) if (tn + fp) > 0 else np.nan
    balanced_accuracy = (sensitivity + specificity) / 2

    balanced_results.append([
        name,
        round(accuracy, 3),
        round(sensitivity, 3),
        round(specificity, 3),
        round(balanced_accuracy, 3)
    ])

# ======================================================
# 6. Performans Tablosu (TABLO 3)
# ======================================================

performance_table = pd.DataFrame(
    performance_results,
    columns=["Model", "Accuracy", "Sensitivity", "Specificity"]
)

print("\n--- TABLO 3:A Model Performans Karşılaştırması ---")
print(performance_table)
balanced_table = pd.DataFrame(
    balanced_results,
    columns=[
        "Model",
        "Accuracy",
        "Sensitivity",
        "Specificity",
        "Balanced Accuracy"
    ]
)

print("\n--- TABLO 3B: Dengeli Model Performansı ---")
print(balanced_table)

import matplotlib.pyplot as plt

# --- Grafik 1: Ham Prevalans ---
prev = df.groupby(['Age_Group', 'Gender_Cat'])['Diagnosis'].mean() * 100
prev = prev.reset_index()

labels = prev['Age_Group'].astype(str) + " - " + prev['Gender_Cat'].astype(str)

plt.figure()
plt.bar(labels, prev['Diagnosis'])
plt.ylabel('Asthma Prevalence (%)')
plt.xlabel('Group')
plt.title('Asthma Prevalence by Age Group and Gender')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("Sekil1_Ham_Prevalans.png")
plt.show()

# --- Grafik 3: Dengeli Modellerde Duyarlılık ---
plt.figure()

plt.bar(
    balanced_table["Model"],
    balanced_table["Sensitivity"]
)

plt.ylabel("Sensitivity")
plt.xlabel("Model")
plt.title("Sensitivity of Balanced Machine Learning Models")

plt.xticks(rotation=30)
plt.tight_layout()
plt.savefig("Sekil3_Dengeli_Modeller_Sensitivity.png")
plt.show()
