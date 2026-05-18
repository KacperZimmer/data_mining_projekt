# Projekt Data Mining: Predykcja podium w wyścigach Formuły 1

## Cel projektu
Budowa i porównanie modeli klasyfikujących (zmienna binarna `top3`) oraz identyfikacja kluczowych czynników sukcesu na podstawie danych historycznych F1.

## Metodologia
CRISP-DM / SEMMA

## Przygotowanie danych
Uruchom skrypt do przygotowania danych wejściowych:
```bash
python data_prep.py
```

Skrypt wczytuje surowe dane z katalogu `f1_data_raw` i wykonuje następujące operacje:
1. Przetworzenie brakujących wartości
2. Stworzenie zmiennej objaśnianej (`top3`)
3. Konsolidacja danych: Połączenie wyników, informacji o wyścigach, kierowcach, konstruktorach i kwalifikacjach
4. Inżynieria cech: m.in. obliczenie wieku kierowcy w dniu wyścigu
5. Wyczyszczenie i uzupełnienie braków danych

Wynikowy zbiór danych znajdziesz w nowo wygenerowanym folderze `f1_data_cleaned` w pliku `f1_processed_dataset.csv`. Będzie to wspólny plik bazowy dla całego zespołu modelarskiego (CART, Random Forest/XGBoost, Sieci Neuronowe).