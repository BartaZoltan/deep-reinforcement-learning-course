# Notebook Template Rules

Ez a fájl a kurzus-notebookok formátum- és generálási szabályait rögzíti.

## 1) Canonical notebook (root, postfix nélkül)

- A **forrás / canonical** notebook mindig a postfix nélküli fájl:
  - példa: `session_02_mdp_dynamic_programming.ipynb`
- Ennek:
  - Colabban futtathatónak kell lennie,
  - lokálisan (Jupyter) futtathatónak kell lennie.

## 2) Kötelező szerkezeti stílus (Session 1 mintára)

- Nyitó blokk:
  - Logo
  - `Developers`, `Date`, `Version`
  - Colab badge
  - `# Practice X: ...`
  - `## Summary`
- Makro felépítés:
  - `##` fő blokkok
  - `### Task N` feladatblokkok
- Pedagógiai sorrend:
  - rövid elmélet -> implementáció -> kísérletek -> interpretáció
- Feladatleírás:
  - mindig konkrét és ellenőrizhető (`What to implement`, `Expected behavior`, `Quick checks` jelleggel)
- Képletek:
  - inline: `$...$`
  - több soros / blokk: `$$...$$`

### 2.1) Kötelező header skeleton (canonical)

Minden canonical notebook eleje kövesse ezt a mintát:

```md
![Logo](...)

**Developers:** ...  
**Date:** YYYY-MM-DD  
**Version:** ...

[<img src="https://colab.research.google.com/assets/colab-badge.svg">](<colab-link>)

# Practice X: <Title>

## Summary

<2-4 mondat rövid összefoglaló>

Content outline:
- ...
- ...
```

Megkötés:
- Ez ne legyen opcionális, hanem minden session notebookban egységesen jelenjen meg.

## 3) Student verzió szabályai

- A student notebook a canonical notebookból készül.
- Fájlnév postfix: `_student.ipynb`.
- A kód-kiszedés marker alapú:
  - két delimiter sor közötti rész kerül blankolásra.
  - delimiter: pontosan `########################################################################` (72 db `#`).
- Elvárás:
  - a delimiter blokk felett legyen markdown vagy komment formában rövid task-instrukció.
- Student verzióban ne maradjon runtime output.

## 4) Web verzió szabályai

- A web notebook a canonical notebookból készül.
- Fájlnév postfix: `_web.ipynb`.
- A `### Task N` markdown cellák eltávolításra kerülnek.
- A code outputokból exportált képek/GIF-ek markdownként beágyazásra kerülnek a webes notebookba.
- A webes assetek célmappája: `assets/web_outputs/`.

## 5) Eredmény-interpretációs szabály (web fókusz)

- Minden fontos eredmény (plot/GIF) alatt legyen interpretáció:
  - mit érdemes megfigyelni,
  - mi a fő tanulság.
- Ez a canonical szerkesztésnél legyen **jelölhető web-only elemmel**.
- Követelmény:
  - a jelölt interpretáció **csak a web verzióban** jelenjen meg,
  - canonical és student verzióban ne jelenjen meg tanulói zavaró extra szövegként.

Javasolt jelölés:
- cell metadata tag: `web_only_explanation`
- vagy egyértelmű marker komment (amit a web-generátor értelmez).

Megjegyzés:
- A generáló scripteknek támogatniuk kell ezt a jelölést.

## 6) Reprodukálhatóság és futtathatóság

- Random seed kezelés legyen explicit.
- A notebook „Run all” után konzisztens eredményt adjon.
- Külső függőség legyen minimális és dokumentált.

## 7) Kötelező záró sanity check (minden canonical notebook végén)

A notebook végén legyen egy rövid sanity-check blokk, ami ellenőrzi:

1. Logikai felépítés:
- a tananyag a könyv szerinti sorrendben halad,
- a taskok egymásra épülnek.

2. Technikai minőség:
- nincs hibás Python kód,
- nincs hibás markdown (képlet/formázás).

3. Kimenetek:
- minden fő eredményhez tartozik értelmezés (különösen webre),
- minden hivatkozott asset (kép/GIF) létezik.

## 8) Naming konvenció összefoglaló

- Canonical: postfix nélküli notebook (root forrás)
- Student: `_student.ipynb`
- Web: `_web.ipynb`
