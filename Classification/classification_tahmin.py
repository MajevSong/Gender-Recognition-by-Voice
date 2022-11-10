import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
from sklearn import model_selection as ms
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

# ------ Logistic Regression ------

# 8. ve 20. kolonlar alınmadı çünkü P value değerleri yüksekti onları çıkarınca R^2 degerinin arttığını gözlemledim
veriler = pd.read_csv("/home/valanis/Desktop/Python/Machine_Learning/datas/voice.csv")
x = veriler.iloc[:, 0:19]
x_drop = x.drop(columns=["kurt"])
y = veriler.iloc[:, -1:]
X = x_drop.values
Y = y.values


x_train, x_test, y_train, y_test = train_test_split(
    X, Y.ravel(), test_size=0.33, random_state=0
)

sc = StandardScaler()

X_train = sc.fit_transform(x_train)
X_test = sc.transform(x_test)

logr = LogisticRegression(random_state=0, max_iter=240)

# Training data y by x
logr.fit(X_train, y_train.ravel())  # type: ignore

# tahmin
y_tahmin_logr = logr.predict(X_test)

cm_logr = confusion_matrix(y_test, y_tahmin_logr)
print(cm_logr)

# Logistic Doğruluk Değeri

dogruluk_degeri_logr = metrics.accuracy_score(y_test, y_tahmin_logr)

print("Sınıflandırma raporu \n", metrics.classification_report(y_test, y_tahmin_logr))

print("logistic temel model dogruluk degeri = {}".format(dogruluk_degeri_logr))

# Cross Validation of Logistic
# Dogruluk degeri tek bir egitim setini baz alarak sonuc cikarir ama
# cross validation yaparsak farklı farklı train-test parcalari alarak
# farklı sonucları elde eder. Bunlarin ortalamasini da alirsak ortalama bir
# dogruluk degerine ulasabiliriz.

basari_logr = cross_val_score(estimator=logr, X=X_train, y=y_train, cv=4)

print(basari_logr.mean())
print(basari_logr.std())
print()

# ------ K-NN ------


# tahmin
knn = KNeighborsClassifier(n_neighbors=5, metric="minkowski").fit(X_train, y_train)

y_tahmin_knn = knn.predict(X_test)

# Accuracy rate of KNN
dogruluk_degeri_knn = metrics.accuracy_score(y_test, y_tahmin_knn)

print("Sınıflandırma raporu \n", metrics.classification_report(y_test, y_tahmin_knn))
print("k-nn temel model dogruluk degeri = {}".format(dogruluk_degeri_knn))

cm_knn = confusion_matrix(y_test, y_tahmin_knn)
print(cm_knn)

# Cross Validation of K-NN

basari_knn = ms.cross_val_score(estimator=knn, X=X_train, y=y_train, cv=4)
print(basari_knn.mean())
print(basari_knn.std())

# ------ SVM ------


# SVC = Support Vector Classification
svc = SVC(kernel="rbf",gamma=0.1,C=5)
svc.fit(X_train, y_train)

y_tahmin_svc = svc.predict(X_test)

cm_svc = confusion_matrix(y_test, y_tahmin_svc)
print(cm_svc)


dogruluk_degeri_svc = metrics.accuracy_score(y_test, y_tahmin_svc)

print("Sınıflandırma raporu \n", metrics.classification_report(y_test, y_tahmin_svc))
print("svc temel model dogruluk degeri = {}".format(dogruluk_degeri_svc))

basari_svc = cross_val_score(estimator=svc, X=X_train, y=y_train, cv=4)
print(basari_svc.mean())
print(basari_svc.std())

# ------ Naive Bayes ------


# bagimsiz ve bagimli degiskenler arasindaki iliskiyi kosullu olasiliga ceviriyoruz

# 1. Tahmin etmek istediginiz veri continuous bir veri ise (reel sayilar, ondalikli sayilar) gaussian kullanilir.

# 2. Veri nominal bir deger ise Multinominal Naive Bayes kullanilir.

# 3. Eger veri binomial deger ise (kadın, erkek gibi) bernouilli naive bayes kullanilir.

# GaussianNB veri setimde BernoulliNB'ye gore daha iyi sonuc verdigi icin onu kullandim.

gnb = GaussianNB()
gnb.fit(X_train, y_train)

y_tahmin_gnb = gnb.predict(X_test)

cm_gnb = confusion_matrix(y_test, y_tahmin_gnb)
print("GNB")
print(cm_gnb)

dogruluk_degeri_gnb = metrics.accuracy_score(y_test, y_tahmin_gnb)

print("Sınıflandırma raporu \n", metrics.classification_report(y_test, y_tahmin_gnb))
print("nvb temel model dogruluk degeri = {}".format(dogruluk_degeri_gnb))

basari_gnb = cross_val_score(estimator=gnb, X=X_train, y=y_train, cv=4)
print(basari_gnb.mean())
print(basari_gnb.std())


# ------ Decision Tree Classification ------


dtc = DecisionTreeClassifier(criterion="entropy")

# X_trainden y_traini ogrenmesini istiyoruz
dtc.fit(X_train, y_train)
y_tahmin_dtc = dtc.predict(X_test)

cm_dtc = confusion_matrix(y_test, y_tahmin_dtc)
print("DTC")
print(cm_dtc)

dogruluk_degeri_dtc = metrics.accuracy_score(y_test, y_tahmin_dtc)

print("Sınıflandırma raporu \n", metrics.classification_report(y_test, y_tahmin_dtc))
print("dtc temel model dogruluk degeri = {}".format(dogruluk_degeri_gnb))

basari_dtc = cross_val_score(estimator=dtc, X=X_train, y=y_train, cv=4)
print(basari_dtc.mean())
print(basari_dtc.std())


# ------ Random Forest Classification ------


rfc = RandomForestClassifier(n_estimators=10, criterion="entropy")
rfc.fit(X_train, y_train)

y_tahmin_rfc = rfc.predict(X_test)

cm_rfc = confusion_matrix(y_test, y_tahmin_rfc)
print("RFC")
print(cm_rfc)

dogruluk_degeri_rfc = metrics.accuracy_score(y_test, y_tahmin_rfc)

print("Sınıflandırma raporu \n", metrics.classification_report(y_test, y_tahmin_rfc))

# Dataset buyuk ise maliyeti düşürmek açısından GridSearchCV yerine RandomSearchCV kullanabiliriz.
# Cunku maliyet crossvalidation ile cok fazla artacaktir.
# GridSearchCV, RandomSearchCV'e gore daha iyi performans gosteren hiperparametre
# setini bulurken, RandomSearchCV bulmayi garanti etmez. Ama RandomSearchCV'in de
# maliyeti GridSearchCV'e gore daha dusuktur.

print("rfc temel model dogruluk degeri = {}".format(dogruluk_degeri_rfc))

# Cross Validation
basari_rfc = cross_val_score(estimator=rfc, X=X_train, y=y_train, cv=4)
print(basari_rfc.mean())
print(basari_rfc.std())

# asagida, manuel olarak hiperparametreyi bulmaya calisiyoruz. Cok fazla denenmesi gereken
# gereken hiperparametre oldugu icin bu sekilde yapilamayacagi acik.
# O yüzden GridsearchCV ya da RandomsearchCV kullanabiliriz.

acc_scores = []
for n in range(1, 11):
    rfc = RandomForestClassifier(min_samples_leaf=n).fit(X_train, y_train)
    y_pred = rfc.predict(X_test)
    acc_scores.append(metrics.accuracy_score(y_test, y_pred))

print(acc_scores)
import matplotlib.pyplot as plt

plt.plot(range(1, 11), acc_scores)
plt.xlabel("min_samples_leaf")
plt.ylabel("Acuracy Scores")
plt.show()

# SVC icin GridsearchCV uyguluyoruz.
# 4 katmanli cross validation yaparak verimizi farkli kernel'lar da deneyerek en iyi optimizasyonu bulmaya calisiyoruz.

# Gama: Düşük gama seçilirse uzak noktalar bile alınırken, yüksek gama da yakın noktalar ele alınır.
# C: Eğitim verilerini ve sınır kararını doğru yapmak için kullanılır. 
p = [
    {
        "C": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        "kernel": ["linear"],
    },
    {
        "C": [1, 2, 3, 4, 5],
        "kernel": ["rbf"],
        "gamma": [1, 0.5, 0.1, 0.01, 0.001],
    },
    {
        "C": [1, 2, 3, 4, 5],
        "kernel": ["poly"],
        "degree": [1, 2, 3, 4, 5, 6, 7],
        "gamma": [1, 0.5, 0.1, 0.01, 0.001],
    },
]

gs = GridSearchCV(estimator=svc, param_grid=p, scoring="accuracy", cv=4)
grid_search = gs.fit(X_train, y_train)

eniyiparamat = grid_search.best_params_
eniyiscore = grid_search.best_score_

print(eniyiparamat)
print(eniyiscore)

# ----- Sonuc ------
# GridSearchCV ile edilen verilere gore;
# Veri setim icin en ideal sınıflandirma algoritmasi: SVC
# SVC Kernel: RBF
# Dogruluk orani: % 98
# Gamma : 0.1
# C: 5

# Bu degerleri en başta SVC'ye yerleştiriyoruz.