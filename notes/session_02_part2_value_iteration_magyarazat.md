# Session 2 - Part 2 magyarázat (Value Iteration)

Kapcsolódó notebook:
- `notebooks/sessions/session_02_mdp_dynamic_programming/session2_mdp_dp_dev.ipynb`

Ez a jegyzet azt foglalja össze, mit érdemes szóban kiemelni a Part 2 (Value Iteration) blokk mellett.

## 1) Fő üzenet

- A Value Iteration egy tervező (planning) algoritmus ismert MDP modellre.
- Nem policyt értékelünk fixen, hanem minden állapotban közvetlenül az optimális Bellman backupot alkalmazzuk.
- A policyt a konvergált értékfüggvényből greedyn nyerjük ki.

## 2) Elméleti minimum

Bellman-optimalitási backup:
$$
V_{k+1}(s)=\max_a\sum_{s',r}p(s',r\mid s,a)\left[r+\gamma V_k(s')\right].
$$

Kiemelendő pontok:
- A `max` lépés miatt implicit policy-improvement történik minden sweepben.
- A frissítés szinkron módon történik ($V_k \rightarrow V_{k+1}$).
- Véges tabuláris MDP-ben konvergál az optimális értékfüggvényhez.

## 3) Implementációs döntések a notebookban

- `bellman_optimality_backup(...)` külön függvényben van a tiszta logikáért.
- Leállás: `delta < theta`, ahol `delta` a sweepen belüli legnagyobb abszolút változás.
- Terminális állapotok explicit kezelése.
- `greedy_actions_from_values(...)` a végső policy kinyerésére.

## 4) Vizualizációk narratívája

- Value heatmap:
  - “Mely állapotok ígéretesek, és mennyire?”
- Greedy policy arrows:
  - “Milyen lokális döntést választ az optimális greedy policy?”
- Convergence curve (`delta`):
  - “Milyen gyorsan csökken a backup-hiba?”
- Snapshotok + GIF:
  - “Hogyan terjed az értékinformáció a célállapot környezetéből?”

## 5) Kísérletek értelmezése

### Gamma sensitivity
- Kérdés: hogyan változik $V(s_0)$ és a policy, ha más a jövő súlya?
- Tipikus minta:
  - nagyobb $\gamma$ hosszabb távú tervezést hangsúlyoz,
  - policy szerkezete változhat kockázatos/stochasztikus környezetben.

### Theta sensitivity
- Kérdés: mennyi számítás kell adott pontossághoz?
- Tipikus minta:
  - kisebb `theta` több sweep és hosszabb futás,
  - de a policy gyakran már lazább küszöbnél is stabil.

### Greedy rollout
- Kérdés: mit csinál a számolt policy tényleges futásban?
- Determinisztikus és slippery összevetésnél jól látszik a környezeti zaj hatása.

## 6) Tipikus kérdések, amikre készülj

“Miért nem ugyanannyi sweep kell mindig?”
- Mert függ a $\gamma$-tól, a stopping thresholdtól, és a környezet dinamikájától.

“Miért lehet hasonló policy eltérő értékek mellett?”
- A greedy akció-rangsor maradhat azonos, miközben az abszolút értékszint változik.

“Miért kell rollout is, ha van value map?”
- A rollout viselkedést mutat, nem csak statikus numerikus becslést.

## 7) Átvezetés Part 3-hoz

- Most már van egy működő Value Iteration baseline.
- Következő lépés: Policy Iteration ugyanazon MDP-n.
- Ezzel összehasonlítható lesz a két klasszikus DP control módszer számítási viselkedése.
