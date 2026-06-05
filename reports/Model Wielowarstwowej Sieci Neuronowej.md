# Model Wielowarstwowej Sieci Neuronowej

## 1. Architektura sieci neuronowej

Ze względu na binarny charakter zmiennej objaśnianej `top3` (1 – ukończenie wyścigu na podium, 0 – poza podium), zaprojektowano model klasyfikacji binarnej oparty o wielowarstwową sieć neuronową (MLP) zaimplementowaną w bibliotece TensorFlow/Keras.

Architektura modelu składa się z następujących elementów:

- **Warstwa wejściowa:** dopasowana do liczby cech po preprocessingu
- **Pierwsza warstwa ukryta (Dense):** 128 neuronów, funkcja aktywacji ReLU, regularyzacja L2 (ograniczenie wag i zapobieganie przeuczeniu)
- **Dropout (0.3):** redukcja przeuczenia poprzez losowe wyłączanie neuronów
- **Druga warstwa ukryta (Dense):** 64 neurony, ReLU
- **Dropout (0.2)**
- **Trzecia warstwa ukryta (Dense):** 32 neurony, ReLU
- **Warstwa wyjściowa:** 1 neuron z funkcją Sigmoid, interpretacja jako prawdopodobieństwo klasy pozytywnej (`top3 = 1`).

Do kompilacji modelu wykorzystano optymalizator **Adam**, funkcję straty **binary_crossentropy** (standardową dla klasyfikacji binarnej) oraz metryki pomocnicze: celność Accuracy i ROC-AUC.

## 2. Preprocessing

Proces trenowania modelu poprzedzono etapem przygotowania danych:

**Feature Engineering**

- Ze zbioru usunięto identyfikatory techniczne oraz zmienną `positionOrder` (która bezpośrednio zdradzała wynik końcowy wyścigu). .
- Po wstępnej budowie oraz wytrenowaniu modelu zastosowano dodatkową inżynierię cech w celu zwiększenia jakości predykcji:
    
    Utworzono nowe zmienne wspomagające, opisujące relacje pomiędzy pozycją startową a wynikiem kwalifikacji oraz charakterystykę startu:
    
    - `grid_vs_quali` **= grid - quali_position**
    
    → różnica pomiędzy pozycją startową a wynikiem kwalifikacji, odzwierciedlająca zmianę pozycji oraz formę zawodnika
    
    - `log_grid` **= log(1 + grid)**
    
    → transformacja logarytmiczna stabilizująca rozkład zmiennej i redukująca wpływ wartości odstających
    
    - `quali_efficiency` **= quali_position / (grid + 1)**
    
    → wskaźnik relacji pomiędzy wynikiem kwalifikacji a pozycją startową
    
    - `grid_advantage` **= grid <= 3**
    
    → zmienna binarna oznaczająca start z pierwszych trzech pozycji, istotnych z punktu widzenia przewagi strategicznej w wyścigu
    

**Podział danych**

Dane podzielono na trzy zbiory z zachowaniem proporcji klas:

- **Treningowy (80%)** – dopasowanie modelu
- **Walidacyjny (10%)** – kontrola procesu uczenia
- **Testowy (10%)** – końcowa, niezależna ewaluacja

**Transformacja danych**

- Zmienne numeryczne (`grid`, `year`, `round`, `circuitId`, `driver_age`, `quali_position`) zostały przeskalowane przy użyciu **StandardScaler**
- Zmienne kategoryczne (`driver_nationality`, `constructor_nationality`) zostały przekształcone algorytmem **OneHotEncoder** do postaci binarnej.

## 3. Proces uczenia i monitorowanie modelu

Model trenowano maksymalnie przez 100 epok z batch size równym 64.

Ze względu na niezbalansowanie klas (dominacja klasy 0 - brak Top 3), zastosowano  technologię wag klas (*class weighting*) - `class_weight = {0: 1, 1: 1.5}.`

Aby zapobiec zjawisku przeuczenia (overfitting), zaimplementowano mechanizm **EarlyStopping** monitorujący funkcję straty na zbiorze walidacyjnym (`val_loss`). 

Poniższe wykresy prezentują przebieg funkcji straty oraz celności w kolejnych epokach:

![Wykres accuracy i loss](../Plots/accuracy_loss.png)

**Wykres celności (Model Accuracy):** Pokazuje procent poprawnych typowań (ok. 88–89%). Wahania wartości na zbiorze walidacyjnym są naturalne i wynikają z losowego charakteru danych oraz nierównowagi klasowej, gdzie klasa pozytywna (Top 3) stanowi mniejszość obserwacji.

**Wykres straty (Model Loss):** Pokazuje, jak szybko sieć uczy się na błędach. Obie linie (niebieska i pomarańczowa) systematycznie spadają, co oznacza, że model działa poprawnie. Ponieważ linia walidacyjna (pomarańczowa) pod koniec nie rośnie, mamy pewność, że model nie jest przeuczony i dobrze poradzi sobie z nowymi danymi.  
Mechanizm **Early Stopping** zakończył proces uczenia w 19 epoce, w momencie gdy dalsze epoki nie przynosiły poprawy na zbiorze walidacyjnym. Pozwoliło to na wybór modelu o najlepszej dostępnej jakości uogólnienia bez ryzyka overfittingu.

## 4. Optymalizacja architektury modelu

W trakcie realizacji projektu przetestowano kilka wariantów architektury sieci neuronowej oraz parametrów uczenia w celu uzyskania jak najlepszych wyników predykcyjnych.

Analizowany był wpływ następujących modyfikacji:

- liczby neuronów w warstwach ukrytych
- wartości współczynnika Dropout
- zastosowania regularyzacji L2
- wag klas (class weighting)
- dodatkowych cech utworzonych w procesie feature engineering
- wartości progu klasyfikacji (threshold)

Następnie porównano uzyskane w każdym modelu wartości Accuracy, Precision, Recall, F1-score oraz ROC-AUC. Największą poprawę jakości modelu zaobserwowano po zastosowaniu dodatkowych cech opisujących relacje pomiędzy pozycją startową a wynikami kwalifikacji oraz po optymalizacji wag klas i progu decyzyjnego.

Ostatecznie wybrano model uzyskujący najlepszy kompromis pomiędzy wykrywaniem przypadków klasy Top 3 (Recall) a trafnością predykcji (Precision).

## 5. Analiza wyników i ocena jakości modelu (Scoring)

Końcowy scoring modelu został przeprowadzony na odłożonym zbiorze testowym. 

Zamiast domyślnego progu 0.5 przeanalizowano różne wartości threshold w celu poprawy jakości klasyfikacji. Celem było znalezienie kompromisu pomiędzy precision (trafnością predykcji) a recall (wykrywalnością klasy Top 3).

Domyślny próg klasyfikacji (0.5) obniżono do wartości 0.4, co pozwoliło uzyskać najlepszy kompromis pomiędzy kluczowymi metrykami jakości, w szczególności precision i recall.

**Wyniki modelu - tabela metryk**

| **Metryka** | **Wartość** |
| --- | --- |
| **Accuracy (Celność)** | 0.8800 |
| **Precision (Precyzja dla klasy 1)** | 0.5239 |
| **Recall (Czułość dla klasy 1)** | 0.6118 |
| **F1-score (dla klasy 1)** | 0.5645 |
| **ROC-AUC** | 0.8927 |

Otrzymane wyniki wskazują, że model charakteryzuje się dobrą ogólną skutecznością predykcji (accuracy ~0.88), jednak ze względu na niezbalansowany charakter danych istotniejsze znaczenie mają metryki precision i recall.

**Precision (~0.52)** oznacza umiarkowaną trafność przewidywań klasy pozytywnej (Top 3)

**Recall (~0.61)** wskazuje, że model jest w stanie poprawnie wykryć większość rzeczywistych przypadków tej klasy. 

Oznacza to, że model częściej poprawnie identyfikuje kierowców kończących w Top 3, kosztem pewnej liczby fałszywych alarmów.

Wysoka wartość **ROC-AUC (~0.89)** świadczy o dobrej zdolności modelu do rozróżniania klas oraz ogólnej jakości predykcyjnej niezależnie od przyjętego progu decyzyjnego.

Wartość **F1-score (~0.56)** potwierdza umiarkowany, ale stabilny kompromis pomiędzy precyzją a czułością modelu.

```
              precision    recall  f1-score   support

           0       0.94      0.92      0.93      2336
           1       0.52      0.61      0.56       340

    accuracy                           0.88      2676
   macro avg       0.73      0.77      0.75      2676
weighted avg       0.89      0.88      0.88      2676
```

**Macierz pomyłek dla modelu predykcji podium F1**

W celu dogłębnej analizy struktury błędów wygenerowano macierz pomyłek (Confusion Matrix)


Macierz przedstawia, że na 2336 przypadków braku podium, model prawidłowo wskazał 2147 z nich (True Negatives). 

W klasie pozytywnej model poprawnie przewidział 208 podiów (True Positives), natomiast popełnił 132 błędy II rodzaju (False Negatives – nie wykrył podium) oraz 189 błędów I rodzaju (False Positives – błędnie wytypował podium).

### 5. Wniosek końcowy

Model wykazuje dobrą zdolność predykcyjną oraz stabilne wyniki w warunkach niezbalansowanych danych. Zastosowanie inżynierii cech oraz optymalizacji progu decyzyjnego poprawiło jakość klasyfikacji, szczególnie w zakresie wykrywania rzadkiej klasy (Top 3).

Jednocześnie relatywnie niska wartość precision (~0.52) wskazuje na istotną liczbę fałszywych pozytywów, co ogranicza jego użyteczność w zastosowaniach wymagających wysokiej precyzji. Model osiąga jednak dobry kompromis pomiędzy precision a recall przy wysokim ROC-AUC (~0.89), co potwierdza jego ogólną skuteczność predykcyjną.
