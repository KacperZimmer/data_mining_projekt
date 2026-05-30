**Porównanie XGBoost z Random Forest**

Porównując XGBoost z wcześniej analizowanym Random Forest, można wskazać
kilka istotnych różnic.

**Skuteczność**

XGBoost uzyskał:

- accuracy na teście: 0.8935

- AUC na teście: 0.8975

Random Forest, w poprzedniej analizie, osiągał:

- accuracy około 0.8842

- AUC około 0.8910

Oznacza to, że XGBoost w tym przypadku okazał się nieco lepszy. Różnica
nie jest ogromna, ale jest zauważalna i przemawia na korzyść
boostingowego podejścia.

Stabilność

XGBoost wykazał bardzo stabilne wyniki cross-validation, wszystkie foldy
były zbliżone. Random Forest również był dobry, ale XGBoost wydaje się
bardziej konsekwentny w rozpoznawaniu wzorców.

Zdolność modelowania zależności

XGBoost, jako metoda boostingowa, lepiej koryguje błędy poprzednich
kroków i często osiąga wyższą jakość predykcji na danych tabelarycznych.
W tym projekcie to właśnie ta cecha najprawdopodobniej pozwoliła uzyskać
lepsze wyniki niż w Random Forest.

Interpretacja cech

W obu modelach najważniejsze były quali_position i grid, ale w XGBoost
ich dominacja była jeszcze wyraźniejsza. To sugeruje, że model
boostingowy lepiej „wyostrzył" zależność między wynikiem kwalifikacji a
szansą na podium.

Wniosek końcowy: który model jest lepszy?

Na podstawie przedstawionych wyników można stwierdzić, że **XGBoost jest
lepszym modelem dla tych danych niż Random Forest**.

Argumenty przemawiające za XGBoost:

1.  Wyższa dokładność testowa
    XGBoost osiągnął 0.8935, podczas gdy Random Forest miał niższą
    skuteczność.

2.  Wyższe AUC
    XGBoost uzyskał 0.8975, czyli lepszą zdolność rozróżniania klas.

3.  Bardziej stabilna walidacja krzyżowa
    Wyniki foldów były bardzo zbliżone, co sugeruje solidną
    generalizację.

4.  Lepsza krzywa uczenia
    Wykres learning curve pokazuje, że model bardzo dobrze uczy się na
    danych treningowych, a przy umiarkowanej liczbie drzew osiąga
    najlepszą jakość na danych walidacyjnych.

5.  Silniejsza identyfikacja najważniejszych cech
    XGBoost jeszcze mocniej podkreślił znaczenie quali_position i grid,
    co jest zgodne z logiką problemu.

Jednocześnie należy zaznaczyć:

Random Forest również był dobrym modelem i stanowił bardzo rozsądną bazę
porównawczą. Jest prostszy, bardziej intuicyjny i zwykle mniej
wymagający w strojenia. Jednak w tym konkretnym zadaniu XGBoost lepiej
wykorzystał sygnał zawarty w danych i zapewnił wyższą jakość predykcji.

Podsumowanie

XGBoost okazał się bardzo skuteczną metodą predykcji miejsc na podium w
wyścigach Formuły 1. Model osiągnął wysoką dokładność i bardzo dobre
AUC, a wykres ROC potwierdził jego zdolność do rozróżniania klas.
Analiza learning curve wskazała, że istnieje optymalna liczba drzew, po
której dalsze zwiększanie złożoności nie poprawia jakości testowej.
Ranking cech pokazał, że kluczowe znaczenie mają przede wszystkim wyniki
kwalifikacji i pozycja startowa, czyli zmienne bezpośrednio związane z
potencjałem sportowym kierowcy.

W porównaniu z Random Forest model XGBoost wypadł korzystniej, zarówno
pod względem skuteczności, jak i stabilności. Z tego względu można
uznać, że dla tego konkretnego zbioru danych i celu predykcyjnego
XGBoost jest rozwiązaniem bardziej efektywnym. Jest to dobry przykład
tego, że choć oba modele należą do rodziny metod opartych na drzewach,
to ich mechanizm działania prowadzi do nieco innych własności, a
boosting często daje przewagę w zadaniach predykcyjnych na danych
tabelarycznych.

Jeśli chcesz, mogę teraz przygotować z tego **wersję formalną do pracy,
bez nagłówków i z bardziej płynnym stylem akademickim**, tak jak
wcześniej dla Random Forest.
