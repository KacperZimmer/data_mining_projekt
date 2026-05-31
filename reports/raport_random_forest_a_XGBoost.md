# Porównanie XGBoost z Random Forest

Porównując XGBoost z wcześniej analizowanym Random Forest, można wskazać kilka istotnych różnic.

## 1. Skuteczność

| Metryka | XGBoost | Random Forest | Lepszy model |
| :--- | :--- | :--- | :--- |
| **Accuracy (Dokładność)** | **0.8935** | 0.8842 | **XGBoost** |
| **AUC** | **0.8975** | 0.8910 | **XGBoost** |

Oznacza to, że XGBoost w tym przypadku okazał się nieco lepszy. Różnica nie jest ogromna, ale jest zauważalna i wyraźnie przemawia na korzyść podejścia opartego na boostingu.

## 2. Kluczowe różnice

* **Stabilność:** XGBoost wykazał bardzo stabilne wyniki walidacji krzyżowej (*cross-validation*), gdzie wszystkie foldy były do siebie zbliżone. Random Forest również radził sobie dobrze, ale XGBoost okazał się bardziej konsekwentny w rozpoznawaniu wzorców.
* **Zdolność modelowania zależności:** XGBoost, jako metoda sekwencyjna (boostingowa), lepiej koryguje błędy popełnione w poprzednich krokach i z reguły osiąga wyższą jakość predykcji na danych tabelarycznych. W tym projekcie to właśnie ta cecha najprawdopodobniej pozwoliła na uzyskanie lepszych rezultatów niż w przypadku lasów losowych.
* **Interpretacja cech:** W obu modelach najważniejsze okazały się zmienne `quali_position` i `grid`, jednak w XGBoost ich dominacja była jeszcze wyraźniejsza. Sugeruje to, że model ten lepiej „wyostrzył” zależność między wynikiem kwalifikacji a ostateczną szansą na podium.

---

## 3. Wniosek końcowy: Który model jest lepszy?

Na podstawie przedstawionych wyników można stwierdzić, że **XGBoost jest lepszym modelem dla tych danych niż Random Forest**.

**Argumenty przemawiające za XGBoost:**

1. **Wyższa dokładność testowa:** Osiągnął wartość 0.8935, podczas gdy Random Forest cechował się nieco niższą skutecznością.
2. **Wyższe AUC:** Uzyskał wynik 0.8975, co oznacza lepszą ogólną zdolność rozróżniania klas.
3. **Bardziej stabilna walidacja krzyżowa:** Zbliżone do siebie wyniki poszczególnych foldów sugerują solidną zdolność uogólniania (generalizacji) modelu.
4. **Lepsza krzywa uczenia:** Wykres *learning curve* dowodzi, że model bardzo sprawnie uczy się na danych treningowych, a najlepszą jakość walidacyjną osiąga już przy umiarkowanej liczbie drzew.
5. **Silniejsza identyfikacja kluczowych cech:** XGBoost jeszcze dobitniej podkreślił znaczenie cech `quali_position` oraz `grid`, co w pełni pokrywa się z logiką problemu badawczego.

> **Warto zaznaczyć:** Random Forest również był dobrym modelem i stanowił rzetelną bazę porównawczą. Jest to algorytm prostszy, bardziej intuicyjny i przeważnie mniej wymagający w kontekście strojenia hiperparametrów. Niemniej jednak, w analizowanym zadaniu XGBoost skuteczniej wykorzystał sygnał zawarty w danych, zapewniając wyższą jakość predykcji.

---

## 4. Podsumowanie

**XGBoost** okazał się wysoce skuteczną metodą predykcji miejsc na podium w wyścigach Formuły 1. Algorytm osiągnął wysoką dokładność oraz bardzo dobre AUC, a kształt krzywej ROC ostatecznie potwierdził jego zdolność do poprawnej separacji klas. 

Analiza krzywej uczenia wskazała na istnienie optymalnej liczby drzew, powyżej której dalsze zwiększanie złożoności nie przekłada się na poprawę jakości testowej. Z kolei ranking ważności cech udowodnił, że kluczowe znaczenie mają wyniki kwalifikacji oraz pozycja startowa — a więc zmienne bezpośrednio odzwierciedlające faktyczny potencjał sportowy kierowcy podczas weekendu wyścigowego.

W bezpośrednim porównaniu z Random Forest, to model **XGBoost wypadł korzystniej** (zarówno pod względem skuteczności, jak i stabilności). Stanowi to doskonały przykład tego, jak mechanizm sekwencyjnego uczenia (boosting) przeważa w zadaniach predykcyjnych opartych na danych tabelarycznych, co ostatecznie czyni XGBoost rozwiązaniem bardziej efektywnym dla niniejszego zbioru danych.