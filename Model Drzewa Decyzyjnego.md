# Model Jednorodnego Drzewa Decyzyjnego (CART)

## 1. Architektura i parametryzacja drzewa decyzyjnego

Ze względu na binarny charakter zmiennej objaśnianej `top3` (1 – ukończenie wyścigu na podium, 0 – poza podium), zaprojektowano model klasyfikacji binarnej oparty o algorytm **CART** (*Classification and Regression Trees*), zaimplementowany w bibliotece `scikit-learn`.

W celu odnalezienia optymalnego punktu równowagi między błędem obciążenia a wariancji oraz dokładnego zbadania zjawiska przeuczenia (*overfittingu*), w procesie badawczym przetestowano trzy niezależne warianty konfiguracyjne algorytmu:

* **Wariant A (Drzewo Pełne):** Model pozbawiony rygorów strukturalnych. Drzewo rozbudowywało się rekurencyjnie aż do momentu osiągnięcia całkowitej czystości węzłów końcowych (brak ograniczenia parametrów `max_depth` oraz `min_samples_leaf`). Model ten stanowił punkt odniesienia do demonstracji przeuczenia na rzeczywistych danych tabularycznych.
* **Wariant B (Drzewo Optymalne z regularyzacją Pre-pruning):** Wariant, w którym zastosowano techniki przycinania wstępnego w celu poprawy zdolności generalizacji. Na podstawie analizy krzywych uczenia nałożono restrykcje: maksymalną głębokość drzewa `max_depth=4` oraz minimalną liczbę obserwacji w liściu `min_samples_leaf=20`. Kryterium podziału węzłów stanowił wskaźnik Giniego.
* **Wariant C (Drzewo Zbalansowane):** Konfiguracja o identycznych parametrach geometrycznych co Wariant B (`max_depth=4`, `min_samples_leaf=20`), lecz z wdrożoną modyfikacją funkcji kosztu za pomocą parametru `class_weight='balanced'`. Wariant ten nadaje wagę proporcjonalnie odwrotną do częstości występowania klas, bezpośrednio przeciwdziałając silnej asymetrii zbioru danych.

## 2. Preprocessing

Proces trenowania modeli poprzedzono wspólnym dla całego zespołu etapem przygotowania i transformacji danych (zgodnie z metodologią laboratoryjną przedmiotu Data Mining):

* **Feature Engineering & Selection:** Do modelu wprowadzono zestaw cech wyselekcjonowanych w ramach projektu: cechy ciągłe/numeryczne (`grid`, `year`, `round`, `circuitId`, `driver_age`, `quali_position`) oraz cechy kategoryczne tekstowe (`driver_nationality`, `constructor_nationality`).
* **Transformacja zmiennych:** Zmienne kategoryczne zostały poddane transformacji zero-jedynkowej przy użyciu `ColumnTransformer` oraz modułu `OneHotEncoder`. Ze względu na występowanie unikalnych, historycznych kategorii narodowościowych w podziale testowym (np. *East German*), koder został zabezpieczony parametrem `handle_unknown='ignore'`. W wyniku kodowania One-Hot przestrzeń cech wejściowych rozrosła się do **72 predyktorów**. Do budowy drzew decyzyjnych nie stosowano standaryzacji cech numerycznych, ponieważ algorytmy oparte na podziałach przestrzeni są niezmiennicze na transformacje monotoniczne predyktorów.
* **Podział zbioru danych:** Zbiór danych został podzielony z zachowaniem proporcji klas (stratyfikacja względem zmiennej celu) na trzy podzbiory: treningowy (80%), walidacyjny (10%) oraz odłożony zbiór testowy (10%), zachowując ziarno losowości `random_state=1`. Końcowa ewaluacja została przeprowadzona na zbiorze testowym o liczebności **2676 obserwacji**.

## 3. Wyniki i ewaluacja modeli

Zgodnie ze standardem oceny klasyfikatorów binarnej zmiennej celu, modele zostały poddane wszechstronnej ewaluacji na zbiorze testowym przy użyciu miar: Celności (Accuracy), Czułości (Sensitivity/Recall), Specyficzności (Specificity) oraz wskaźnika AUC (Area Under ROC Curve).

**Tabela 1. Zestawienie zbiorcze metryk dla wariantów modelu drzewiastego (CART)**

| Wariant Modelu | Accuracy | Czułość (Sensitivity) | Specyficzność (Specificity) | AUC Score |
| :--- | :---: | :---: | :---: | :---: |
| **Wariant A (Pełne)** | 0.8506 | 0.4475 | 0.9093 | 0.6784 |
| **Wariant B (Optymalne)** | **0.8884** | 0.3582 | **0.9655** | **0.8905** |
| **Wariant C (Zbalansowane)** | 0.7899 | **0.8528** | 0.7807 | **0.8878** |

### Szczegółowa analiza raportów klasyfikacji (Wariant B vs Wariant C)

Warianty zoptymalizowane (B i C) osiągnęły bardzo wysokie wskaźniki **AUC (odpowiednio 0.8905 oraz 0.8878)**, co świadczy o wysokiej i stabilnej zdolności rozdzielczej modeli drzewiastych na danych tabularycznych Formuły 1. Kluczowe różnice ujawniają się jednak w rozkładzie błędów I i II rodzaju:

* **Wariant B (Optymalny – zorientowany na globalną celność):**
  Model ten osiągnął najwyższą ogólną celność (**88.84%**) oraz doskonałą specyficzność (**96.55%**). Bardzo rzadko popełnia błąd typu I (False Positive) – rzadko błędnie typuje podium dla kierowcy, który go nie zdobędzie. Przekłada się to na precyzję (Precision) dla klasy pozytywnej na poziomie 60%. Główną wadą jest jednak niska czułość (**35.82%**) – model działa wysoce zachowawczo i pomija wiele rzeczywistych podiów.

```text
               precision    recall  f1-score   support

Brak Podium (0)       0.91      0.97      0.94      2336
   Podium (1)       0.60      0.36      0.45       340

     accuracy                           0.89      2676
    macro avg       0.76      0.66      0.69      2676
 weighted avg       0.87      0.89      0.88      2676
```

* **Wariant C (Zbalansowane – zorientowany na wykrywanie podium):**
  Poprzez automatyczną korektę wag klas, model drastycznie podniósł czułość dla klasy pozytywnej aż do poziomu **85.28%** (błędy typu II zostały zminimalizowane). Skutkuje to jednak spadkiem specyficzności do 78.07% i częstszym generowaniem fałszywych alarmów (błędów I rodzaju), co obniża precyzję klasy pozytywnej do 36% oraz globalne Accuracy do 78.99%.

```text
               precision    recall  f1-score   support

Brak Podium (0)       0.97      0.78      0.87      2336
   Podium (1)       0.36      0.85      0.51       340

     accuracy                           0.79      2676
    macro avg       0.67      0.82      0.69      2676
 weighted avg       0.90      0.79      0.82      2676
```

## 4. Macierz pomyłek i analiza błędów

W celu bezpośredniego porównania struktur decyzyjnych w warunkach asymetrii klas (w próbie testowej znalazło się 2336 przypadków braku podium oraz 340 przypadków ukończenia wyścigu w TOP 3; łączna liczebność próby $N = 2676$), przeanalizowano liczbowe macierze pomyłek:

* **Macierz pomyłek – Wariant B (Optymalne):**
  * **True Negatives (Prawidłowy brak podium):** 2255 obserwacji
  * **False Positives (Błędne wytypowanie podium – Błąd I rodzaju):** 81 obserwacji
  * **False Negatives (Pominięte podium – Błąd II rodzaju):** 218 obserwacji
  * **True Positives (Prawidłowo wykryte podium):** 122 obserwacje

* **Macierz pomyłek – Wariant C (Zbalansowane):**
  * **True Negatives (Prawidłowy brak podium):** 1824 obserwacje
  * **False Positives (Błędne wytypowanie podium – Błąd I rodzaju):** 512 obserwacji
  * **False Negatives (Pominięte podium – Błąd II rodzaju):** 50 obserwacji
  * **True Positives (Prawidłowo wykryte podium):** 290 obserwacji

Z perspektywy merytorycznej, wybór między modelami zależy od bezpośredniego celu stajni wyścigowej. Wariant B jest idealnym narzędziem konserwatywnym minimalizującym ryzyko błędnego wytypowania (gdy prognozuje podium, precyzja wynosi aż 60%). Wariant C z kolei idealnie nadaje się do szerokiego filtrowania kandydatów, bezbłędnie eliminując kierowców niemających żadnych szans na TOP 3 (tylko 50 pominiętych podiów na 2676 startów).

### Graficzna reprezentacja macierzy pomyłek

![Rysunek 1. Macierz pomyłek dla Wariantu B (Optymalnego) na odłożonej próbie testowej.](Plots/macierz_b.png)

*Rysunek 1. Macierz pomyłek dla Wariantu B (Optymalnego) na odłożonej próbie testowej.*

![Rysunek 2. Macierz pomyłek dla Wariantu C (Zbalansowanego) na odłożonej próbie testowej.](Plots/macierz_c.png)

*Rysunek 2. Macierz pomyłek dla Wariantu C (Zbalansowanego) na odłożonej próbie testowej.*

## 5. Wizualizacja geometryczna struktury drzewa

W celu interpretacji logicznej dokonanych przez model podziałów, poniżej przedstawiono graficzny schemat wygenerowanego drzewa klasyfikacyjnego dla optymalnego Wariantu B:

![Rysunek 3. Graficzna struktura podziałów przestrzeni cech dla optymalnego drzewa decyzyjnego.](Plots/struktura_drzewa_cart.png)

*Rysunek 3. Graficzna struktura podziałów przestrzeni cech dla optymalnego drzewa decyzyjnego.*

### Wniosek końcowy w kontekście reguł merytorycznych

Analiza tekstu reguł decyzyjnych oraz powyższej wizualizacji graficznej dla najlepszego modelu drzewiastego pozwoliła na bezpośrednie odkrycie wiedzy ekonomiczno-sportowej:

1. **Priorytet Kwalifikacji:** Korzeń drzewa decyzyjnego dokonuje pierwszego podziału na podstawie cechy `quali_position <= 6.50` połączonej bezpośrednio z `grid <= 3.50`. Algorytm matematycznie udowodnił wyścigową regułę mówiącą, że start z pierwszych trzech pozycji drastycznie odseparowuje prawdopodobieństwo sukcesu od reszty stawki.
2. **Ewolucja Technologiczna:** Pojawienie się w strukturze progów historycznych (np. `year <= 1987.50`) udowadnia, że drzewo decyzyjne doskonale zinterpretowało ewolucję Formuły 1. W nowoczesnej erze (wysoka bezawaryjność bolidów), wysoka pozycja startowa jest niemal bezpośrednim gwarantem dowiezienia podium, podczas gdy w latach 70. i 80. gigantyczna losowość mechaniczna silników niszczyła przewagę uzyskaną w kwalifikacjach.