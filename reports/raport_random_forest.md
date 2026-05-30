1. Cel projektu i charakter zadania

Celem projektu było zbudowanie modelu uczenia maszynowego, który
przewiduje, czy kierowca Formuły 1 ukończy wyścig na podium, czyli w
pierwszej trójce (top3). Jest to typowe zadanie klasyfikacji binarnej,
ponieważ zmienna docelowa przyjmuje jedynie dwie wartości: 1, jeśli
kierowca znalazł się na podium, oraz 0, jeśli nie osiągnął takiego
wyniku.

Tego rodzaju problem ma duże znaczenie praktyczne, ponieważ pozwala
analizować, które czynniki historyczne i sportowe zwiększają szansę na
sukces. W sporcie motorowym wynik wyścigu zależy od wielu zmiennych
jednocześnie: pozycji startowej, formy kierowcy, jakości bolidu, toru,
sezonu, a także pośrednio od doświadczenia zawodnika. Z tego powodu
zastosowanie modelu uczącego się z danych historycznych jest uzasadnione
i pozwala wykrywać wzorce, które nie zawsze są oczywiste przy zwykłej
analizie statystycznej.

2. Random Forest jako model złożony

Do rozwiązania zadania zastosowano Random Forest, czyli las losowy. Jest
to metoda należąca do grupy modeli zespołowych, zwanych również ensemble
methods. W modelach zespołowych nie polega się na jednym klasyfikatorze,
lecz na połączeniu wielu prostszych modeli, których wyniki są agregowane
w końcową decyzję.

2.1. Teoria drzew decyzyjnych

Random Forest bazuje na drzewach decyzyjnych. Drzewo decyzyjne dzieli
przestrzeń cech na coraz mniejsze regiony poprzez serię pytań o wartość
zmiennych. Na każdym węźle model wybiera cechę i próg podziału, które
najlepiej rozdzielają obserwacje na klasy. Ostatecznie każda obserwacja
trafia do liścia, gdzie przypisywana jest klasa.

Drzewa decyzyjne są intuicyjne, łatwe do interpretacji i dobrze radzą
sobie z danymi nieliniowymi. Mają jednak poważną wadę: są podatne
na przeuczenie. Oznacza to, że mogą bardzo dobrze dopasować się do
danych treningowych, ale gorzej generalizować na nowe dane. W praktyce
pojedyncze drzewo bywa zbyt wrażliwe na przypadkowe odchylenia w
zbiorze.

2.2. Bagging i losowość w Random Forest

Random Forest rozwiązuje ten problem poprzez połączenie idei baggingu i
losowości cech. Bagging, czyli bootstrap aggregating, polega na
trenowaniu wielu modeli na losowo dobranych próbkach danych z
powtórzeniami. Każde drzewo widzi nieco inny wariant zbioru uczącego, co
zwiększa różnorodność modeli.

Dodatkowo przy każdym podziale drzewa losowo wybiera się tylko część
cech, spośród których wybierany jest najlepszy podział. Dzięki temu
drzewa nie są do siebie zbyt podobne. Ostateczna predykcja jest
uzyskiwana przez głosowanie większościowe w klasyfikacji. W rezultacie:

- zmniejsza się wariancja modelu,

- poprawia się stabilność predykcji,

- rośnie odporność na szum,

- maleje ryzyko przeuczenia.

Random Forest jest więc przykładem modelu złożonego, który łączy wiele
słabszych elementów w silniejszy, bardziej odporny system predykcyjny. W
kontekście danych o wyścigach Formuły 1 jest to bardzo dobre podejście,
ponieważ wynik zależy od wielu wzajemnie oddziałujących czynników.

3. Przygotowanie danych i podział zbioru

W projekcie wykorzystano dane zapisane w pliku CSV. Zawierały one
informacje o wyścigach, kierowcach, zespołach, kwalifikacjach, pozycji
startowej, torze, roku i innych cechach opisujących kontekst
rywalizacji. Zmienna docelowa top3 informowała, czy dany start zakończył
się miejscem na podium.

3.1. Usunięcie potencjalnego leakage

Wcześniej usunięto kolumnę positionOrder, ponieważ jest to cecha
bezpośrednio związana z końcowym wynikiem wyścigu. Jej pozostawienie
mogłoby prowadzić do wycieku informacji. W uczeniu maszynowym leakage
występuje wtedy, gdy model korzysta z danych, które w rzeczywistym
momencie predykcji nie byłyby jeszcze znane. W takim przypadku wynik
modelu byłby sztucznie zawyżony i nie miałby wartości praktycznej.

3.2. Kodowanie cech kategorycznych

Zmienne tekstowe, takie jak narodowość kierowcy i konstruktora, zostały
zakodowane numerycznie. Jest to konieczne, ponieważ algorytmy uczenia
maszynowego operują na danych liczbowych. W projekcie
użyto LabelEncoder, który przypisuje każdej kategorii unikalny kod
liczbowy. Jest to rozwiązanie proste i wystarczające dla modelu Random
Forest, ponieważ ten algorytm nie interpretuje kodów jako wartości
porządkowych w taki sam sposób jak modele liniowe.

3.3. Podział danych

Zastosowano podział na:

- zbiór treningowy: (21407, 12), 80%

- zbiór walidacyjny: (2676, 12), 10%

- zbiór testowy: (2676, 12). 10%

Po oddzieleniu zmiennej docelowej:

- X_train: (21407, 11),

- X_val: (2676, 11),

- X_test: (2676, 11).

Takie podejście jest bardzo dobre metodologicznie, ponieważ pozwala:

1.  wytrenować model na największym zbiorze danych,

2.  dostroić i ocenić go na walidacji,

3.  przeprowadzić końcowy, niezależny pomiar jakości na zbiorze
testowym.

Podział na trzy części pozwala odróżnić ocenę bieżącą od finalnej i
ogranicza ryzyko nadmiernego dopasowania do jednego konkretnego zbioru.

4. Rozkład klas i jego znaczenie

W analizowanym zbiorze występuje wyraźna nierównowaga klas.

Zbiór treningowy:

- 0: 18689 przypadków,

- 1: 2718 przypadków.

Zbiór walidacyjny:

- 0: 2337 przypadków,

- 1: 339 przypadków.

Zbiór testowy:

- 0: 2336 przypadków,

- 1: 340 przypadków.

Oznacza to, że przypadki podium stanowią znacznie mniejszościową klasę.
To bardzo typowa sytuacja w problemach predykcyjnych, gdzie wydarzenia
pozytywne są rzadsze niż negatywne. W praktyce oznacza to, że model może
mieć tendencję do faworyzowania klasy 0, dlatego sama accuracy nie
wystarczy do pełnej oceny jakości. Przy nierównowadze klas dużo
ważniejsze staje się także AUC, które lepiej pokazuje zdolność modelu do
rozróżniania klas.

5. Wyniki modelu i ich interpretacja

Po wytrenowaniu Random Forest uzyskano następujące wyniki:

- Dokładność na zbiorze walidacyjnym: 0.8815

- Dokładności cross-validation: [0.8821, 0.8832, 0.8911, 0.8918,
  0.8883]

- Średnia dokładność cross-validation: 0.8873

- AUC na zbiorze walidacyjnym: 0.8810

- Dokładność na zbiorze testowym: 0.8842

- AUC na zbiorze testowym: 0.8910

5.1. Accuracy

Accuracy oznacza odsetek poprawnie sklasyfikowanych przypadków.
Wartość 0.8815 na walidacji i 0.8842 na teście oznacza, że model
poprawnie klasyfikował około 88% obserwacji. Jest to dobry rezultat,
szczególnie w zadaniu z nierównowagą klas. Warto jednak zauważyć, że
accuracy może być częściowo wspierana przez przewagę klasy 0, dlatego
nie powinna być jedynym wskaźnikiem jakości.

5.2. Walidacja krzyżowa

Wyniki 5-krotnej cross-validation są bardzo stabilne i zbliżone do
siebie. Wszystkie wartości mieszczą się w przedziale około 0.882--0.892,
a średnia wynosi 0.8873. To bardzo dobry znak, ponieważ oznacza, że
model zachowuje podobną skuteczność niezależnie od konkretnego podziału
danych.Taka stabilność sugeruje, że model rzeczywiście nauczył się
ogólnych wzorców, a nie tylko zapamiętał konkretne przykłady.

5.3. AUC

Wartość AUC na poziomie 0.8810 dla walidacji i 0.8910 dla testu wskazuje
na bardzo dobrą zdolność rozróżniania klas. AUC, czyli Area Under the
Curve, mierzy pole pod krzywą ROC i jest jedną z najważniejszych metryk
dla klasyfikatorów binarnych. Wartość bliska 0.9 oznacza, że model
bardzo dobrze odróżnia przypadki pozytywne od negatywnych.

6. Krzywa ROC i jej znaczenie

![](./media/image1.png){width="6.3in" height="5.034722222222222in"}

Krzywa ROC przedstawia kompromis między:

- True Positive Rate --- odsetkiem poprawnie wykrytych przypadków
  pozytywnych,

- False Positive Rate --- odsetkiem przypadków negatywnych błędnie
  sklasyfikowanych jako pozytywne.

Im bardziej krzywa ROC zbliża się do lewego górnego rogu wykresu, tym
lepszy model. Na przedstawionym wykresie krzywa Random Forest znajduje
się wyraźnie powyżej przekątnej klasyfikatora losowego, co oznacza, że
model przewiduje znacząco lepiej niż przypadek. Nie jest to jednak model
idealny, co jest pozytywne z punktu widzenia wiarygodności wyników ---
rezultaty nie wyglądają sztucznie.

AUC na poziomie około 0.89 oznacza, że jeśli wylosujemy losowo jeden
przypadek pozytywny i jeden negatywny, model w około 89% przypadków
przypisze wyższe prawdopodobieństwo klasie pozytywnej. Jest to bardzo
dobry wynik i potwierdza skuteczność modelu.

7. Ważność cech i ich interpretacja

Random Forest umożliwia ocenę feature importance, czyli ważności cech.
Pokazuje to, które zmienne najbardziej przyczyniały się do decyzji
modelu.

Najważniejsze cechy w modelu były następujące:

1.  quali_position --- 0.171555

2.  grid --- 0.152466

3.  driver_age --- 0.125134

4.  raceId --- 0.116952

5.  circuitId --- 0.085365

6.  round --- 0.080892

7.  year --- 0.070867

8.  driverId --- 0.069476

9.  constructorId --- 0.065401

10. driver_nationality --- 0.041207

11. constructor_nationality --- 0.020685

7.1. quali_position

To najważniejsza cecha modelu. Jest to całkowicie zgodne z intuicją
sportową, ponieważ wynik kwalifikacji bardzo silnie wpływa na szanse
zdobycia podium. Kierowca startujący z wysokiej pozycji ma większe
szanse na utrzymanie czołowej lokaty.

7.2. grid

Pozycja startowa również jest kluczowa. W sportach motorowych start z
przodu daje przewagę strategiczną i zmniejsza ryzyko utknięcia za
wolniejszymi zawodnikami. Na wielu torach, zwłaszcza trudnych do
wyprzedzania, pozycja startowa jest jednym z najważniejszych czynników
sukcesu.

7.3. driver_age

Wiek kierowcy może być interpretowany jako pośrednia miara doświadczenia
i etapu kariery. Młodsi kierowcy mogą mieć większą dynamikę, starsi ---
większe doświadczenie. Model wykrył, że ta cecha ma istotne znaczenie
predykcyjne.

7.4. raceId, circuitId, round, year

Te cechy nie odnoszą się bezpośrednio do umiejętności kierowcy, ale
opisują kontekst wyścigu. Ich ważność sugeruje, że niektóre tory, rundy
i sezony sprzyjają lepszym wynikom niż inne. Możliwe jest również, że
model wykorzystuje te zmienne jako pośrednie wskaźniki formy zespołów w
określonych momentach sezonu.

7.5. driverId i constructorId

Identyfikatory kierowcy i zespołu mają znaczenie, ponieważ silni
zawodnicy i konkurencyjne zespoły częściej walczą o podium. Model może w
ten sposób „rozpoznawać" kierowców i konstruktorów, którzy historycznie
osiągali lepsze wyniki.

7.6. driver_nationality i constructor_nationality

Te cechy miały najmniejszy wpływ na predykcję. Oznacza to, że same
narodowości nie są kluczowym czynnikiem sukcesu. Ich znaczenie jest
raczej pośrednie i mniejsze niż cech związanych bezpośrednio z formą
sportową.

8. Wnioski metodologiczne

Przeprowadzona analiza pokazuje, że Random Forest jest skutecznym i
sensownym wyborem dla tego typu zadania. Model:

- dobrze radzi sobie z danymi tabelarycznymi,

- osiąga wysoką skuteczność,

- zapewnia stabilne wyniki,

- umożliwia interpretację ważności cech.

Najważniejsze jest jednak to, że wyniki są realistyczne i spójne. W
odróżnieniu od wcześniejszych prób, tutaj nie obserwujemy nienaturalnego
perfekcjonizmu modelu. Accuracy, AUC i cross-validation wskazują na
dobry, ale nie idealny poziom skuteczności, co jest dużo bardziej
wiarygodne w rzeczywistym problemie predykcyjnym.

Pod względem merytorycznym projekt potwierdza, że:

- pozycja w kwalifikacjach i pozycja startowa są najważniejszymi
  czynnikami wpływającymi na podium,

- cechy kontekstowe, takie jak tor, runda i sezon, również mają
  znaczenie,

- narodowość ma niewielki wpływ w porównaniu z cechami stricte
  sportowymi.

9. Podsumowanie końcowe

Zbudowany model Random Forest pozwolił skutecznie przewidywać
prawdopodobieństwo zajęcia miejsca na podium w wyścigach Formuły 1.
Wysokie wartości accuracy i AUC, a także stabilne wyniki walidacji
krzyżowej, wskazują, że model posiada dobrą zdolność generalizacji.
Największe znaczenie miały cechy związane bezpośrednio z przebiegiem
wyścigu i formą kierowcy, przede wszystkim quali_position i grid.

Projekt pokazuje również, jak istotne jest właściwe przygotowanie
danych, eliminacja leakage oraz stosowanie kilku metryk oceny
jednocześnie. Dzięki temu można nie tylko zbudować działający model, ale
także zrozumieć, dlaczego działa on w określony sposób i które czynniki
rzeczywiście wpływają na wynik.
