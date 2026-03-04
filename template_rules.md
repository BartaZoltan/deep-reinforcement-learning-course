# Notebook Template Rules

Ez a dokumentum a kurzus-notebookok forrás- és generálási szabályainak egységes specifikációja.

## 1) Source Of Truth

- A canonical notebook mindig a postfix nélküli fájl.
- Példa: `session_02_mdp_dynamic_programming.ipynb`.
- A canonical verzió futtatható kell legyen:
  - lokálisan (Jupyter),
  - Google Colabban.
- A `_student` és `_web` fájlok generált artefaktok, kézzel nem szerkesztjük őket.

## 2) Kötelező notebook szerkezet

- Nyitó blokk minden sessionben azonos stílusban:
  - logo,
  - `Developers`, `Date`, `Version`,
  - Colab badge,
  - `# Practice X: ...`,
  - `## Summary` (rövid cél + tartalom).
- Fő címek: `##`.
- Feladatblokkok: `### Task N`.
- Pedagógiai sorrend:
  - elméleti alap,
  - implementáció,
  - kísérlet,
  - rövid interpretáció.
- Képletek:
  - inline: `$...$`,
  - blokk: `$$...$$`.

## 3) Task formátum (kötelező)

Minden task leírása ezt a formátumot kövesse:

1. `### Task N`
2. üres sor
3. `**Task name (X min)**`
4. rövid, konkrét leírás arról, mit kell implementálni és mit kell ellenőrizni.

Követelmény:
- A leírás alapján a hallgató képes legyen a delimiter közötti kódot önállóan megírni.
- A leírás ne legyen túl „machine-like”, de legyen technikailag pontos.

## 4) Delimiter és student generálás

- A megoldás-blokkokat a canonical notebookban markereli:
  - `########################################################################` (pontosan 72 db `#`).
- A `_student.ipynb` generáláskor a delimiter párok közti kód törlődik/blankelődik.
- A blokk fölött mindig legyen task-instrukció markdownban vagy kommentben.
- A student notebookban runtime output nem marad.

## 5) Web generálás és output beágyazás

- A `_web.ipynb` a canonical notebookból készül.
- A `### Task N` markdown cellák kikerülnek.
- Minden releváns vizuális outputot markdownként be kell ágyazni közvetlenül a forrás code cella után.
- Támogatott források:
  - notebook inline image output (`image/png`, `image/jpeg`, `image/gif`),
  - szöveges outputban hivatkozott mentett asset (`Saved GIF/PNG: assets/web_outputs/...`).
- Web asset célmappa: `assets/web_outputs/`.
- A web generátor másolja át a mentett asseteket a website content alá is.

## 6) Colab badge szabály

- Minden notebook badge-je a saját fájljára mutasson.
- Ez kötelező canonical, student, web és homework verziókra is.
- Minta:
  - `https://colab.research.google.com/github/BartaZoltan/deep-reinforcement-learning-course/blob/main/<notebook-path>`

## 7) Reproducibility

- Seed kezelés legyen explicit.
- `Run all` után reprodukálható, konzisztens kimenetet adjon.
- Külső dependency legyen minimális és dokumentált.

## 8) Eredmény-interpretáció szabály

- Minden fontos ábra/GIF után legyen rövid magyarázó markdown.
- Ne csak azt írja le, „mi látszik”, hanem a fő tanulságot is.
- A szöveg rövid, de informatív legyen.

## 9) Kötelező sanity check a notebook végén

A canonical notebook végén legyen ellenőrző blokk, ami validálja:

1. logikai sorrend (könyv szerinti ív, taskok egymásra épülése),
2. technikai minőség (nincs hibás Python/markdown),
3. output integritás (hivatkozott képek/GIF-ek léteznek).

## 10) Naming konvenció

- Canonical: `<session_name>.ipynb`
- Student: `<session_name>_student.ipynb`
- Web: `<session_name>_web.ipynb`
- Homework (ha van): `<session_name>_homework.ipynb`

## 11) Ajánlott generálási workflow

1. Canonical notebook frissítése és futtatása.
2. Student verzió generálása.
3. Web verzió generálása.
4. Website sync futtatása (`scripts/sync_notebooks_to_website.py`).
5. Gyors ellenőrzés:
   - Colab badge self-linkek,
   - beágyazott outputok megjelennek,
   - website build nem veszít el GIF-eket.
