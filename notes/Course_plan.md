Reinforcement Learning Gyakorlati Kurzus (14 alkalom)

Bevezető

A kurzus célja, hogy a diákok PhD‑szintű, mégis gyakorlatorientált megértést szerezzenek a megerősítéses tanulás (RL) különböző módszereiről. A foglalkozások 1,5 órásak, Python/PyTorch környezetben futtatható Jupyter‑füzeteket használunk, hogy a hallgatók minden algoritmust kódolni és kipróbálni tudjanak. A Szaton és Barto „Reinforcement Learning: An Introduction” könyv fejezeteire építünk, ugyanakkor a legújabb kutatási eredményeket is integráljuk.

Megjegyzés: Az első alkalom a multi‑armed bandit problémákról már kidolgozásra került, így az alábbi terv a 2–14. alkalmakhoz javasol tananyagot.

2. alkalom – Markov döntési folyamatok és dinamikus programozás
	•	Elméleti háttér: Markov döntési folyamatok (MDP), értékfunkciók, Bellman‑egyenletek, policy értékelés és policy iteráció, értékiteráció.
	•	Gyakorlat:
	•	Egyszerű rácsos környezetben (Gridworld) MDP definíció és értékelés írása.
	•	Python‑ban implementálni a policy‑iterációs és értékiterációs algoritmust.
	•	Paraméterek (γ, θ) szerepének vizsgálata.
	•	Kapcsolódó források: Sutton könyv 4. fejezet; a jegyzetekben néhány feladathoz a FrozenLake környezetet is használhatjuk.

3. alkalom – Monte Carlo módszerek
	•	Elméleti háttér: Első és minden‑látogatásos Monte Carlo értékbecslés, on‑policy Monte Carlo control, felfedezési indulások, importance sampling.
	•	Gyakorlat:
	•	Blackjack környezet (OpenAI Gym) használatával Monte Carlo értékbecslő írása.
	•	On‑policy vs. off‑policy Monte Carlo control összehasonlítása különböző felfedezési stratégiákkal.
	•	Kísérletek arról, hogyan hat a minta mérete és az epsilonszint az eredményre.
	•	Kapcsolódó források: Sutton könyv 5. fejezet.

4. alkalom – Időbeli differencia módszerek (TD‑learning)
	•	Elméleti háttér: TD(0) értékbecslés, SARSA és Q‑learning algoritmus; epsilon‑greedy felfedezés; összehasonlítás Monte Carlo módszerekkel.
	•	Gyakorlat:
	•	WindyGridworld vagy CliffWalking környezetre TD(0) és SARSA implementálása.
	•	Q‑learning implementáció FrozenLake környezetben.
	•	Hyperparaméterek (tanulási ráta, ε) hatásának vizsgálata.
	•	Kapcsolódó források: Sutton könyv 6. és 7. fejezet.

5. alkalom – Többlépéses visszatérések és jogosultsági nyomok
	•	Elméleti háttér: n‑lépéses TD, n‑lépéses SARSA, TD(λ) és eligibility trace (jogosultsági nyomok).
	•	Gyakorlat:
	•	N‑lépéses SARSA implementációja MountainCar környezetben, paraméterek (n) vizsgálata.
	•	TD(λ) algoritmus kódolása, λ hatásának összehasonlítása a tanulás sebességére.
	•	Kapcsolódó források: Sutton könyv 7. és 12. fejezet; a diákok megérthetik, hogy a n‑lépéses visszatérések hogyan közelítik meg a TD(0) és a Monte Carlo módszerek végletét.

6. alkalom – Funkcióközelítés és lineáris értékbecslés
	•	Elméleti háttér: Lineáris funkcióközelítés, felügyelt tanulás és RL kapcsolata, állapotjellemzők (feature engineering), nagydimenziós állapotterek kezelése.
	•	Gyakorlat:
	•	CartPole vagy MountainCar környezetben lineáris funkcióközelítővel TD(0) és SARSA implementálása.
	•	Összehasonlítás tabuláris és approximált megoldások között.
	•	Esetleg tanulók által javasolt jellemzők létrehozása.
	•	Kiegészítő: Rövid bevezetés a Fourier és tile coding módszerekbe.

7. alkalom – Deep Q‑Network (DQN)
	•	Elméleti háttér: Mély Q‑hálózat alapjai, célhálózat és tapasztalati tár (experience replay), paraméteres Q‑háló tanítása. A DQN algoritmus sikere, ugyanakkor az értékek túlbecslésének problémája.
	•	Gyakorlat:
	•	DQN implementálása PyTorch‑ban a CartPole v0 vagy LunarLander v2 környezetben.
	•	Replay buffer használatának kódolása.
	•	Megtapasztalható, hogy a célt hálózatos frissítés stabilizálja a tanulást.
	•	Kiegészítő: A DQN túlbecslési hiba csökkentésére a Double Q‑tanulás motivációja és a következő alkalmakon tárgyalt fejlesztések bevezetése.

8. alkalom – DQN fejlesztések: Double DQN, dueling architektúra, prioritásos replay és Rainbow
	•	Double DQN: A Double Q‑tanulás az értékek túlbecslésének csökkentésére jött létre. A Double DQN‑papír kimutatta, hogy a DQN algoritmus bizonyos Atari játékoknál komoly túlbecslési hibát szenved, ezért a szerzők egy olyan módosítást javasoltak, amely a maximális cselekvés kiválasztását és értékelését szétválasztja ￼.
	•	Dueling DQN: A dueling háló két párhuzamos ágat tanít: egy állapotérték és egy cselekvési előny (advantage) ágat. Ez által jobban generalizál az akciók között, és az Atari tesztekben új állapot‑of‑the‑art eredményt hozott ￼ ￼.
	•	Prioritized Experience Replay (PER): A tapasztalatok uniform újrajátszása helyett a fontos átmenetek gyakrabban kerülnek kiválasztásra; a PER‑rel bővített DQN a vizsgált 49 játékból 41‑ben felülmúlta a hagyományos DQN‑t ￼.
	•	Rainbow DQN: A Rainbow algoritmus hat különböző DQN‑fejlesztést – többek között Double DQN, dueling architektúra, prioritásos replay, több lépéses visszatérés, disztribúciós értékbecslés és NoisyNet felfedezést – egyesített. A tanulmány kimutatta, hogy ezek kombinációja új state‑of‑the‑art teljesítményt eredményez az Atari 2600 benchmarkon, mind adat‑hatékonyságban, mind végső pontszámban ￼.
	•	Gyakorlat:
	•	Double DQN implementálása a korábbi DQN kód módosításával.
	•	Dueling háló és prioritásos replay implementálása a replay buffer módosításával.
	•	A hallgatók kis csoportokban dolgozhatnak Rainbow funkciók beépítésén – a GPU nélküli környezet miatt kisebb, mégis nem triviális környezeteket (pl. LunarLander vagy Pong‑ram) használunk.

9. alkalom – Policy gradient módszerek: REINFORCE és Actor–Critic
	•	Elméleti háttér: Policy‐based vs. value‐based tanulás; REINFORCE algoritmus, baseline használata; determinisztikus és stochasztikus politikák; az actor–critic keretrendszer, ahol az actor frissíti a politikát és a kritikus értékbecslést végez.
	•	Gyakorlat:
	•	REINFORCE implementálása egyszerű folyamatos akciójú környezetben (pl. Pendulum vagy MountainCarContinuous).
	•	Advantage Actor–Critic (A2C) implementálása, ahol a policy gradienshez advantage‑becslést használunk.
	•	Összehasonlítás a Q‑alapú módszerekkel.

10. alkalom – Fejlett policy gradient: Aszinkron és proximal algoritmusok
	•	Aszinkron Advantage Actor–Critic (A3C): A DeepMind 2016‑os tanulmánya javasolta az aszinkron tanulók használatát; az A3C a több szálon futó actor‑learner egységek stabilizáló hatása miatt GPU nélkül is versenyképes. A legjobban teljesítő módszer, az aszinkron actor‑critic, felülmúlta a korábbi state‑of‑the‑art eredményeket és csak egy CPU‑t igényelt ￼.
	•	Trust Region Policy Optimization (TRPO): A TRPO algoritmus iteratív eljárást ad a politika optimalizálására, garantált monoton javulással; a szerzők szerint nagyméretű neurális hálók optimalizálására is alkalmas, és a tesztekben robusztus teljesítményt mutatott különböző robotikus mozgásfeladatokon ￼.
	•	Proximal Policy Optimization (PPO): A PPO család új, egyszerűen implementálható policy‑gradient módszereket vezetett be; egy „surrogate” objektívval több mini‑batch frissítést tesz lehetővé, és empirikusan jobb minta‑hatékonyságot nyújt, mint a hagyományos politikagradiens módszerek ￼.
	•	Gyakorlat:
	•	A3C/A2C vs. PPO implementálása. A3C‑t futtathatjuk CPU‑n, de a futtatási idő miatt demonstráció formájában is elegendő.
	•	PPO implementálása a LunarLanderContinuous környezetben, paramétertuninggal.
	•	Feladat: hasonlítsák össze a tanulási görbéket TRPO‑t utánzó beállításokkal.

11. alkalom – Off‑policy actor–critic: DDPG, TD3 és Soft Actor‑Critic
	•	Elméleti háttér:
	•	DDPG (Deep Deterministic Policy Gradient): determinisztikus policy és target hálók kombinációja, alkalmas folyamatos akcióterekre.
	•	TD3 (Twin Delayed DDPG): kettős kritikus háló a túlbecslés csökkentésére, késleltetett frissítések.
	•	Soft Actor‑Critic (SAC): Maximális entrópiájú célfüggvény; az aktor nemcsak a várható jutalmat, hanem az entrópiát is maximalizálja. A Soft Actor‑Critic off‑policy algoritmus, amely a maximum entrópiás keretrendszer alapján definiálja a célt; a szerzők azt találták, hogy az SAC stabil és minta‑hatékony, és különböző folyamatos irányítási feladatokon felülmúlta a többi módszert ￼.
	•	Gyakorlat:
	•	DDPG implementálása saját replay bufferrel és célt hálóval; feladat a Pendulum v0 kitartó szabályozása.
	•	TD3 implementálása a DDPG kiterjesztésével; a diákok összehasonlíthatják a túlbecslés és a teljesítmény különbségeit.
	•	Soft Actor‑Critic implementálása (kisebb környezetben, pl. Pendulum vagy BipedalWalker).
	•	Szükség esetén Google Colab használata GPU‑val.

12. alkalom – Intrinszik motiváció és felfedezés
	•	Intrinszik jutalmazás: A nagy állapotterekben vagy ritka jutalmazású problémákban a kíváncsiság segíthet. A Curiosity‑driven Exploration by Self‑supervised Prediction tanulmány a kíváncsiságot úgy definiálja, hogy az agent által okozott állapotváltozások előrejelzési hibáját jutalmazza ￼, és kimutatta, hogy ez a módszer sparse reward környezetekben (pl. VizDoom, Super Mario Bros.) hatékonyabb felfedezést eredményez.
	•	Random Network Distillation (RND): Az RND egy fix random hálózat által generált reprezentációk előrejelzési hibáját használja intrinzik jutalomként. Az RND bónusz lehetővé tette, hogy a Montezuma’s Revenge Atari játékban az agent state‑of‑the‑art teljesítményt érjen el anélkül, hogy demonstrációkat használt volna ￼.
	•	Gyakorlat:
	•	Egyszerű rácsos környezetben (pl. háromszobás labirintus) RND beépítése DQN‑be.
	•	Alternatív megoldás: egy gridworld, ahol nincs külső jutalom, a hallgatók az intrinzik jutalmazás révén tanulnak elérni egy célállapotot.

13. alkalom – Model‑based RL és világmodellek
	•	Elméleti háttér: Model‑based RL áttekintése: környezetmodell tanulása, tervezés (planning) és tanulás kombinálása, pl. Dyna‑stílusú architektúrák; world model, latent imagination.
	•	Dreamer: A Dreamer algoritmus a tanult világmodellt latent térben futtatott imaginációval kombinálja. A szerzők szerint a Dreamer képes hosszú horizontú feladatok megoldására pusztán képekből, mivel a tanult állapotértékek gradiensét visszavezetik a világmodell által képzelt trajektóriákon ￼.
	•	Gyakorlat:
	•	Egyszerű modell‑based próbálkozások: Dyna‑Q kisebb rácsos környezetben (modell építése az átmenetekből, majd a modell segítségével generált szimulált tapasztalatok felhasználása).
	•	Haladó opció: a Dreamer kódjának futtatása (csak demonstráció), vagy world model tanítása egyedibb, kisméretű környezeten (pl. cartpole images) CPU‑val/Colab‑al.
	•	Megbeszéljük a modellalapú RL előnyeit és kihívásait.

14. alkalom – Haladó témák: disztribúciós RL, offline RL, curriculum learning és RLHF
	•	Disztribúciós RL: A „Distributional Perspective on RL” dolgozat a visszatérés teljes eloszlásának becslését javasolja, nem csak a várható értékét. A szerzők disztribúciós Bellman egyenletet vezetnek be, és az algoritmusuk az Atari játékokon állapot‑of‑the‑art eredményeket ért el ￼.
	•	State‑of‑the‑art: Rainbow DQN (összefoglalás a 8. alkalom alapján).
	•	Offline RL röviden: Az offline RL célja, hogy kizárólag előre rögzített adatból tanuljon; áttekintjük a koncepciót, például konzervatív Q‑tanulást (CQL) vagy BCQ, és megmutatjuk, milyen problémákat okoz a disztribúciós eltérés.
	•	Curriculum learning: A 2025‑ös „Automating Curriculum Learning” cikk kiemeli, hogy a curriculum (egyszerű feladatoktól a nehezebbek felé haladó tanulási folyamat) csökkenti a tanulási időt és robusztusabb politikákat eredményez. A szerzők egy Skill‑Environment Bayesian Network‑et javasolnak, amely különböző feladatok nehézségét modellezi és a megfelelő feladatok kiválasztását automatizálja ￼ ￼.
	•	RLHF (Reinforcement Learning from Human Feedback): Rövid ismertető arról, hogyan használható emberi visszajelzés a jutalmazási függvény becslésére. A diákok számára opcionális feladat: felhasználói preferenciák gyűjtése egy egyszerű környezetben (pl. videójáték), majd preferencia‑optimalizáció.
	•	Záróprojekt: Minden hallgató válasszon egy state‑of‑the‑art cikket (pl. distributional RL, SAC, Dreamer, curriculum learning, RLHF) és készítsen Jupyter‑füzetet, melyben röviden bemutatja az algoritmust, implementálja egy egyszerű környezetben és prezentálja az eredményeit.

Megjegyzések a kurzus kivitelezéséhez
	•	Környezeti könyvtárak: A kurzus a gymnasium/gym könyvtár (vagy PettingZoo ha több ügynös környezet kellene), NumPy és PyTorch használatára épít. A legújabb könyvtárverziók tanácsosak (pl. gymnasium>=0.28).
	•	Hardver: A legtöbb feladat CPU‑n is futtatható. Az Atari és SAC/TD3 feladatoknál érdemes Google Colab GPU‑t használni, de az egyszerű környezetekhez nincs szükség GPU‑ra.
	•	Jegyzetek és füzetek: Minden alkalomra készítsünk Jupyter‑füzetet, amely tartalmaz:
	•	Rövid elméleti összefoglalót.
	•	Kódsejteket kiindulási kerettel (pl. algoritmus class skeleton, környezet inicializálás).
	•	Gyakorló feladatokat, ahol a hallgatóknak kitöltendő részek vannak.
	•	Reflexiós kérdéseket, pl. „Mi történik, ha növeled az n‑lépéses visszatérés lépésnagyságát?”
	•	Értékelés: a hallgatók a heti feladatokat rövid leírással és az elért tanulási görbékkel adják le. A záróprojekt prezentációval zárul.

A fenti tematika rugalmasan alakítható a hallgatók érdeklődése szerint. A klasszikus tananyag (Sutton könyv) és a legújabb RL‑kutatások kombinációja biztosítja, hogy a kurzus naprakész gyakorlati tudást adjon át.