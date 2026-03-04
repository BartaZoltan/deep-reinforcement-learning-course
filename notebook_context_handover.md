# Notebook Context Handover

Ez a fájl rövid, gyakorlati kontextus-összefoglaló a kurzus-notebookok szerkezetéről, a generálási elvekről, és a jelenlegi állapotról. Célja, hogy egy új chat / másik instance gyorsan fel tudja venni a fonalat.

## 1) A notebookok szerepe

A notebookok a Deep Reinforcement Learning kurzus gyakorlati anyagai. Minden session egy önálló, futtatható Jupyter notebook, amely:

- rövid elméleti bevezetést ad,
- feladatokra bontja az implementációt,
- kísérleteket és vizualizációkat tartalmaz,
- a Sutton & Barto könyv megfelelő fejezeteire épül,
- lokálisan és Colabban is futtatható kell legyen.

A cél nem csak a végső megoldás bemutatása, hanem az algoritmusok felépítése, megértése és összehasonlítása.

## 2) Három notebook-verzió logika

Minden sessionhez több notebook-verzió tartozhat.

### Canonical notebook

- Ez a fő forrás.
- Postfix nélküli fájl.
- Ezt szerkesztjük kézzel.
- Ennek futtathatónak kell lennie lokálisan és Colabban.

Példa:

- `notebooks/sessions/2_mdp_dynamic_programming/mdp_dynamic_programming.ipynb`

### Student notebook

- A canonical notebookból generált verzió.
- A delimiter (`########################################################################`) közti megoldás-kód törlődik / blankelődik.
- Runtime output nem marad benne.

Példa:

- `.../mdp_dynamic_programming_student.ipynb`

### Web notebook

- A canonical notebookból generált, publikációs célú verzió.
- A `### Task N` markdown cellák kikerülnek.
- A vizuális outputok markdownként beágyazódnak a kódcellák után.
- A webes megjelenítéshez használt assetek `assets/web_outputs/` alá kerülnek.

Példa:

- `.../mdp_dynamic_programming_web.ipynb`

## 3) Kötelező notebook-stílus

A Session 1 és Session 2 szolgál mintának. A fejléc egységes.

Kötelező nyitó blokk:

- logo,
- `Developers`,
- `Date`,
- `Version`,
- Colab badge,
- `# Practice X: ...`,
- `## Summary`.

Az általános szerkezet:

- `##` fő blokkcímek,
- `### Task N` feladatblokkok,
- elmélet -> implementáció -> kísérlet -> rövid interpretáció.

Matematikai jelölések:

- inline: `$...$`
- blokk: `$$...$$`

## 4) Task-formátum

Minden feladatleírás ezt a mintát kövesse:

1. `### Task N`
2. üres sor
3. `**Task name (X min)**`
4. konkrét, emberileg követhető leírás

Fontos:

- a hallgató a leírás alapján elvileg meg tudja írni a delimiter közötti hiányzó részt,
- a leírás ne legyen túl rövid és ne legyen túl gépies,
- ne csak „implementáld ezt”, hanem adjon elég kontextust a célról és az elvárt viselkedésről.

## 5) Delimiter szabály

A canonical notebookban a megoldásblokkok jelölése:

- `########################################################################`

Ez pontosan 72 darab `#`.

Ezt a student-generátor használja arra, hogy a megoldást kiszedje.

## 6) Colab-kompatibilitás

Minden notebook Colab badge-je a saját fájljára mutasson.

Ez igaz:

- canonical,
- student,
- web,
- homework

verziókra is.

Az egyik fontos tanulság: a notebookok ne feltételezzenek fix working directory-t. Ha helyi helper fájl (`utils.py`) kell, akkor:

- több lehetséges pathon próbáljuk keresni,
- ha nincs meg, Colabban / lokálisan is lehessen GitHub raw URL-ről letölteni.

## 7) Utility-k használata

Visszatérő helper függvényeknél az irány az, hogy ne minden session notebook tartalmazzon nagy mennyiségű duplikált kódot.

Két szint hasznos:

### Session-local utils

- például `notebooks/sessions/3_monte_carlo_methods/utils.py`
- ide kerülhetnek a session-specifikus vizualizációk, egyszerű helper függvények, asset-kezelés

### Később shared utils

- egy közös `notebooks/shared_utils/` mappa jó irány lehet
- ide mennének az általánosabban újrahasznosítható helper függvények

Jelenlegi elv:

- ami még csak egy sessionhöz kell, maradhat session-szinten `utils.py`-ban,
- amit több session is használ, később érdemes közös modulba szervezni.

## 8) Website / web build tanulság

A website build nem a kézzel szerkesztett `_web.ipynb` fájlt tekinti elsődleges igazságnak, hanem a CI:

- `scripts/sync_notebooks_to_website.py`

scriptje generálja újra a website tartalmat a canonical notebookokból.

Ezért ha valami a webes deploy során „visszaáll”, akkor jellemzően ezt a scriptet kell javítani, nem csak a web notebookot.

### Fontos javítás, ami már bekerült

A `scripts/sync_notebooks_to_website.py` most már nem csak a közvetlen notebook image outputokat tudja beágyazni, hanem az olyan szöveges outputból is felismeri és átmásolja a hivatkozott asseteket, mint:

- `Saved GIF: assets/web_outputs/...`
- `Saved PNG: assets/web_outputs/...`

Tehát a GitHub Pages build már képes a mentett GIF/PNG asseteket is:

- átmásolni `website/content/assets/web_outputs/` alá,
- beágyazni a web notebook megfelelő markdown celláiba.

Ez kritikus a Session 2-höz, ahol sok vizualizáció nem inline image-ként, hanem mentett fájlként jelenik meg.

## 9) Jelenlegi session-nevezési konvenció

A session mappák át lettek nevezve az új rövid formára:

- `1_k_armed_bandit`
- `2_mdp_dynamic_programming`
- `3_monte_carlo_methods`
- ...
- `14_advanced_topics`

A notebookfájlok is topic-alapú rövid nevet kaptak.

Példák:

- `notebooks/sessions/1_k_armed_bandit/k_armed_bandit.ipynb`
- `notebooks/sessions/2_mdp_dynamic_programming/mdp_dynamic_programming.ipynb`
- `notebooks/sessions/3_monte_carlo_methods/monte_carlo_methods.ipynb`

Ez azt jelenti, hogy a notebookfájl neve már nem egyezik automatikusan a session mappa nevével.

Emiatt a website sync scriptet módosítani kellett: most már automatikusan megkeresi az egyetlen canonical notebookot a session mappában ahelyett, hogy a mappanévből próbálná kitalálni a fájlnevet.

## 10) Session 2 (MDP / DP) állapot

A Session 2 erősen kidolgozott állapotban van.

Fő elemek:

- saját GridWorld implementáció,
- Value Iteration,
- Policy Iteration,
- nagyobb mapok,
- FrozenLake bemutatás,
- Gambler’s Problem,
- sok GIF / PNG vizualizáció,
- web/student generálási logika finomítva.

Több fontos vizualizáció már mentett assetként működik, nem feltétlen inline outputként.

## 11) Session 3 (Monte Carlo) jelenlegi állapot

A Session 3 jelenleg fejlesztés alatt áll.

Jelenlegi fájlok:

- `notebooks/sessions/3_monte_carlo_methods/monte_carlo_methods.ipynb`
- `notebooks/sessions/3_monte_carlo_methods/utils.py`

### Mi készült el eddig

A notebook eleje már fel van építve ugyanazzal a header-stílussal, mint Session 1 és 2.

Beépített blokkok:

1. MC prediction bevezetés
- First-Visit MC
- Every-Visit MC

2. FrozenLake alap demo
- 4x4 deterministic
- 4x4 slippery

3. Task 1
- episode generation fix policy alatt

4. Task 2
- First-Visit és Every-Visit MC prediction implementáció

5. Prediction kísérletek
- value heatmap összehasonlítás
- sample-size sensitivity

6. Task 3
- On-policy MC control epsilon-greedy policy improvementtel

7. Control kísérletek
- deterministic vs slippery policy comparison
- epsilon sensitivity
- episode-budget sensitivity

8. Task 4
- Off-policy MC prediction
- ordinary importance sampling
- weighted importance sampling

9. Off-policy kísérlet
- behavior epsilon hatása

### Mi fontos a Session 3-mal kapcsolatban

- A fő környezet jelenleg FrozenLake, mert jól kapcsolódik a Session 2-höz.
- A notebook egyszerű, érthető formában készül, kevés absztrakcióval.
- A vizuális helper függvények a session-local `utils.py`-ban vannak.

### A `utils.py` jelenlegi szerepe Session 3-ban

Tartalmaz:

- seed beállítás,
- opcionális letöltő helper (`download_if_missing`),
- FrozenLake map render,
- state-value heatmap,
- side-by-side value comparison,
- policy nyíl-grid megjelenítés,
- egyszerű curve/bar plot helper függvények.

## 12) Fontos technikai buktatók

### `FileNotFoundError` utils betöltésnél

Ez korábban előjött, mert a notebook fix relatív úton próbálta importálni a `utils.py`-t.

Megoldás:

- több útvonalat próbálunk,
- ha nincs meg, GitHub raw-ról letöltjük,
- ez Colabban és lokálisan is működjön.

### Web notebookban „nem látszanak a gifek”

Ennek tipikus okai:

1. a GIF csak `Saved GIF: ...` szövegként szerepel a cell outputban,
2. a web-generátor nem másolja át a fájlt,
3. a web-generátor nem embedeli a fájlhivatkozást markdownként.

Ez a website sync scriptben már kezelve van.

## 13) Ajánlott következő lépések Session 3-hoz

1. Tovább finomítani a Monte Carlo notebookot:
- rendes futtatható kimenetek,
- web/student verziók generálása

2. Következő blokk:
- off-policy rész finomítása,
- vagy további environment (pl. Blackjack) hozzáadása

3. A Session 4 (TD methods) felé átvezetés:
- hangsúlyozni, hogy MC teljes epizódokra vár,
- noisy környezetben lassú és nagy varianciájú,
- ezért lesz indokolt a bootstrapping / TD.

## 14) Mit érdemes megadni egy új instance-nak rövid promptként

Ha ezt a teljes fájlt nem akarod bemásolni, akkor röviden ezek a legfontosabb kontextusok:

1. A canonical notebook a postfix nélküli forrás; a student és web notebook generált artefakt.
2. A taskok `### Task N` + `**Task name (X min)**` formában mennek, a megoldásblokkokat a 72 db `#` delimiter jelöli.
3. A website CI a `scripts/sync_notebooks_to_website.py` scriptből generál mindent, ezért a webes viselkedést ott kell javítani.
4. A session mappák új neve rövid forma: `1_k_armed_bandit`, `2_mdp_dynamic_programming`, `3_monte_carlo_methods`, stb.
5. A notebookfájlok topic-alapú rövid nevet kaptak (`k_armed_bandit.ipynb`, `mdp_dynamic_programming.ipynb`, `monte_carlo_methods.ipynb`).
6. Session 3 már tartalmazza a FrozenLake-alapú First/Every-Visit MC prediction, on-policy MC control és off-policy MC prediction blokkokat.
7. Session 3 helper függvényei jelenleg a `notebooks/sessions/3_monte_carlo_methods/utils.py` fájlban vannak.

