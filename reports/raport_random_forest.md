# Projekt: Predykcja miejsc na podium w Formule 1 (Model Random Forest)

## 1. Cel projektu i charakter zadania

Celem projektu było zbudowanie modelu uczenia maszynowego, który przewiduje, czy kierowca Formuły 1 ukończy wyścig na podium (w pierwszej trójce – `top3`). 

Zadanie to jest typowym przykładem **klasyfikacji binarnej**, ponieważ zmienna docelowa przyjmuje wyłącznie dwie wartości:
* **1** – kierowca znalazł się na podium.
* **0** – kierowca nie osiągnął takiego wyniku.

W sporcie motorowym wynik wyścigu zależy od wielu połączonych zmiennych: pozycji startowej, formy kierowcy, jakości bolidu, charakterystyki toru, momentu w sezonie, a także doświadczenia zawodnika. Zastosowanie modelu uczącego się z danych historycznych pozwala analizować skuteczność tych czynników i wykrywać wzorce, które nie są oczywiste przy tradycyjnej analizie statystycznej.

---

## 2. Random Forest jako model złożony

Do rozwiązania zadania zastosowano algorytm **Random Forest (Las Losowy)**. Jest to model zespołowy (*ensemble method*), który zamiast opierać się na jednym klasyfikatorze, łączy wiele prostszych modeli (drzew decyzyjnych), aby wygenerować ostateczną decyzję.

### 2.1. Teoria drzew decyzyjnych
Drzewo decyzyjne dzieli przestrzeń cech na coraz mniejsze regiony poprzez serię pytań o wartość zmiennych. Na każdym węźle model wybiera cechę i próg podziału najlepiej rozdzielające obserwacje na klasy. 
* **Zalety:** Są intuicyjne, łatwe do interpretacji i dobrze radzą sobie z danymi nieliniowymi. 
* **Wady:** Są silnie podatne na przeuczenie (*overfitting*) – mogą zapamiętywać dane treningowe zamiast uczyć się ogólnych reguł.

### 2.2. Bagging i losowość w Random Forest
Random Forest neutralizuje wady pojedynczych drzew za pomocą dwóch mechanizmów:
1. **Bagging (Bootstrap Aggregating):** Każde drzewo jest trenowane na losowo dobranej próbie danych z powtórzeniami, co zwiększa różnorodność całego "lasu".
2. **Losowość cech:** Przy każdym podziale węzła, algorytm wybiera najlepszą cechę z losowo ograniczonego podzbioru zmiennych, zapobiegając nadmiernemu upodabnianiu się drzew.

Ostateczna predykcja to wynik głosowania większościowego. W efekcie model zmniejsza wariancję, poprawia stabilność predykcji, staje się odporny na szum i minimalizuje ryzyko przeuczenia. To doskonałe narzędzie do analizy złożonych interakcji między czynnikami wpływającymi na wyścig F1.

---

## 3. Przygotowanie danych i podział zbioru

Wykorzystano zbiór danych (CSV) zawierający informacje m.in. o wyścigach, kierowcach, zespołach, kwalifikacjach, pozycji startowej i torach. 

### 3.1. Usunięcie potencjalnego przecieku informacji (Data Leakage)
Usunięto kolumnę `positionOrder` (końcowa pozycja), ponieważ jest bezpośrednio skorelowana z wynikiem. Jej pozostawienie doprowadziłoby do sztucznego zawyżenia wyników modelu, odbierając mu jakąkolwiek wartość predykcyjną.

### 3.2. Kodowanie cech kategorycznych
Zmienne tekstowe (np. narodowość kierowcy/konstruktora) zakodowano numerycznie przy pomocy `LabelEncoder`. Ponieważ Random Forest operuje na podziałach, nie ma ryzyka błędnej interpretacji wartości liczbowych jako wartości porządkowych (co bywa problemem w modelach liniowych).

### 3.3. Podział danych
Dane podzielono według klasycznego schematu:

| Zbiór | Wymiary | Proporcja | Rola |
| :--- | :--- | :--- | :--- |
| **Treningowy (X_train)** | (21407, 11) | 80% | Trening modelu |
| **Walidacyjny (X_val)** | (2676, 11) | 10% | Strojenie hiperparametrów i bieżąca ocena |
| **Testowy (X_test)** | (2676, 11) | 10% | Niezależna, finalna ocena skuteczności |

---

## 4. Rozkład klas i zjawisko nierównowagi

W zbiorze występuje wyraźna i naturalna dla tego problemu nierównowaga klas:

| Zbiór | Klasa 0 (Brak podium) | Klasa 1 (Podium) |
| :--- | :--- | :--- |
| **Treningowy** | 18 689 | 2 718 |
| **Walidacyjny** | 2 337 | 339 |
| **Testowy** | 2 336 | 340 |

Większość przypadków to sytuacje, w których kierowca nie stanął na podium. W takich warunkach metryka *accuracy* (dokładność) bywa myląca (model przewidujący zawsze "0" miałby wysoką dokładność). Dlatego równolegle analizowano metrykę **AUC** (Area Under the ROC Curve).

---

## 5. Wyniki modelu i ich interpretacja

| Metryka | Walidacja | Test | Cross-Validation (Trening) |
| :--- | :--- | :--- | :--- |
| **Accuracy (Dokładność)** | 0.8815 | **0.8842** | Średnia: **0.8873** |
| **AUC** | 0.8810 | **0.8910** | Wyniki foldów: [0.8821, 0.8832, 0.8911, 0.8918, 0.8883] |

### 5.1. Accuracy
Dokładność rzędu 88.4% oznacza, że model poprawnie sklasyfikował większość obserwacji. Jest to bardzo dobry wynik.

### 5.2. Walidacja krzyżowa (CV)
Wyniki 5-krotnej walidacji krzyżowej (0.882–0.892) są bardzo stabilne. Taka niska wariancja między "foldami" potwierdza zdolność modelu do uogólniania wzorców i brak nadmiernego dopasowania do konkretnego fragmentu danych.

### 5.3. AUC
Wynik na poziomie blisko 0.89 to znak, że model bardzo sprawnie rozróżnia przypadki pozytywne od negatywnych, pomimo nierównowagi klas. 

---

## 6. Krzywa ROC

![Krzywa ROC - Random Forest](./media/image1.png) *(Upewnij się, że grafika znajduje się w odpowiednim katalogu)*

Wykres ten przedstawia kompromis między odsetkiem trafnych wykryć (*True Positive Rate*) a odsetkiem fałszywych alarmów (*False Positive Rate*). 

Krzywa Random Forest znajduje się wyraźnie powyżej przekątnej losowego klasyfikatora. Wartość AUC (0.89) udowadnia, że w 89% przypadków wylosowania jednej pary wyników pozytywny/negatywny, model prawidłowo przypisze wyższe prawdopodobieństwo przypadkowi pozytywnemu.

---

## 7. Ważność cech (Feature Importance)

Ranking cech wpływających na decyzje Random Forest:

| Miejsce | Cecha | Ważność | Znaczenie w modelu |
| :--- | :--- | :--- | :--- |
| **1** | **`quali_position`** | **0.1716** | Wynik z kwalifikacji (najważniejszy wskaźnik potencjału wyścigowego). |
| **2** | **`grid`** | **0.1525** | Pozycja startowa (kluczowa strategia i przewaga na torze). |
| 3 | `driver_age` | 0.1251 | Doświadczenie vs. dynamika kierowcy. |
| 4 | `raceId` | 0.1170 | Kontekst sezonowy (specyfika wyścigu). |
| 5 | `circuitId` | 0.0854 | Charakterystyka toru. |
| 6 | `round` | 0.0809 | Moment sezonu. |
| 7 | `year` | 0.0709 | Postęp technologiczny, zasady danego sezonu. |
| 8 | `driverId` | 0.0695 | Umiejętności/renoma konkretnego zawodnika. |
| 9 | `constructorId` | 0.0654 | Budżet i siła zespołu konstruktorów. |
| 10 | `driver_nationality` | 0.0412 | Marginalne znaczenie kontekstowe. |
| 11 | `constructor_nationality`| 0.0207 | Najmniej istotna cecha. |

### Interpretacja:
Zgodnie z intuicją, **wyniki z kwalifikacji i pozycja startowa dominują w predykcji**. Model poprawnie zidentyfikował również, że zmienne biograficzne (np. wiek kierowcy) mają znaczenie w kontekście doświadczenia, ale już narodowość kierowcy czy konstruktora to zmienne o charakterze niemal marginalnym.

---

## 8. Podsumowanie metodologiczne i końcowe

Zbudowany model **Random Forest** okazał się skutecznym i logicznym wyborem do tego typu analizy:
* Doskonale radzi sobie z danymi tabelarycznymi.
* Jest odporny na nierównowagę klas, generując stabilne i wysokie wyniki AUC.
* Umożliwia precyzyjne śledzenie wpływu poszczególnych cech.

Kluczowe w projekcie było profesjonalne podejście do przygotowania danych (wykluczenie zjawiska wycieku informacji) oraz zastosowanie rygorystycznego podziału testowego (CV, Walidacja, Test). Skuteczność na poziomie ok. 88-89% pokazuje realistyczne odwzorowanie dynamiki wyścigów F1, dowodząc, że za wynik odpowiada głównie twardy potencjał sportowy ukształtowany podczas kwalifikacji i pozycji na gridzie.