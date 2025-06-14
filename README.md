# **Izvještaj o projektu: AI agent za edukaciju o ulaganju**

---

### **1. Naslov i članovi tima**

**Naslov:**
AI agent za edukaciju o ulaganju

**Članovi tima:**

* Leo Vukoje
* Franko Šojat

---

### **2. Opis teme i odabrane domene**

Cilj projekta je razviti AI agenta za edukaciju o osnovama ulaganja i financijske pismenosti. Agent koristi lokalne LLM modele i vlastitu bazu edukativnih materijala (.txt i .pdf) te odgovara na pitanja korisnika vezana uz dionice, obveznice, fondove, štednju, rizike i druge financijske pojmove.
Domena je namijenjena početnicima i svima koji žele naučiti više o ulaganju na jasan, pristupačan i siguran način.

---

### **3. Opis tehničke arhitekture**

**Arhitektura uključuje sljedeće module:**

* **Korisničko sučelje:** Streamlit aplikacija (chat + pregled sesija + vizualizacija embeddinga)
* **Preprocessing:** Parsiranje .txt i .pdf edukativnih dokumenata, chunkanje teksta, generiranje embeddinga (HuggingFace sentence-transformers)
* **Vektorska baza:** ChromaDB za pohranu embeddinga i retrieval
* **LLM sloj:** Lokalni model (Mistral-7B-Instruct Q4\_K\_M .gguf) pokrenut s llama-cpp-python
* **Logiranje:** Zapis svake korisničke interakcije u JSON log (za analizu i evaluaciju)
* **Vizualizacija embeddinga:** t-SNE 2D prikaz s Plotly-jem
* **Evaluacija:** Ručna i automatska analiza točnosti odgovora

**Protok podataka:**

1. Korisnik postavlja pitanje kroz Streamlit.
2. Pitanje se pretvara u embedding, traže se najsličniji chunkovi u bazi.
3. Top-k chunkova se šalje kao “kontekst” lokalnom LLM-u.
4. LLM generira odgovor koji se prikazuje korisniku i zapisuje u log.

**Korištene biblioteke:**

* langchain, chromadb, sentence-transformers, llama-cpp-python, pypdf, streamlit, scikit-learn, plotly, matplotlib

---

### **4. Opis korištenih alata i modela**

**Alati:**

* **Streamlit** – za izradu interaktivnog chat sučelja i pregleda sesija/logova
* **Python** – backend logika
* **Jupyter Notebook** (za evaluaciju i vizualizaciju)

**Modeli:**

* **Mistral-7B-Instruct Q4\_K\_M (.gguf)** – lokalni LLM, dovoljno brz za laptop/desktop bez GPU-a
* **sentence-transformers/all-MiniLM-L6-v2** – embedding model za vektorizaciju tekstova

---

### **5. Prikaz i analiza evaluacije**

**Evaluacija modela provedena je na 15-ak tipičnih pitanja iz domene (npr. “Što je investicijski fond?”, “Koji su rizici ulaganja?”, “Kako prepoznati financijsku prijevaru?”).**

* Ručna evaluacija (ocjene, komentari) prikazana je u tablici niže.
* Uz ručnu evaluaciju, provedena je i **napredna automatska evaluacija** koristeći tri metrike: **BLEU**, **ROUGE** i **BERTScore**.

#### **Kratki opis korištenih metrika:**

* **BLEU/ROUGE:** Automatske metrike koje mjere sličnost između generiranog i očekivanog odgovora na temelju preklapanja riječi i fraza.
* **BERTScore:** Napredna metrika koja koristi duboke jezične modele za ocjenu semantičke sličnosti između generiranog i idealnog odgovora.

---

#### **Rezultati automatske evaluacije (za 5 najtipičnijih pitanja):**

| **Pitanje**                                 | **BLEU** | **ROUGE-1** | **ROUGE-L** | **BERTScore F1** |
| ------------------------------------------- | -------- | ----------- | ----------- | ---------------- |
| Što je investicijski fond?                  | 0.05     | 0.28        | 0.28        | 0.78             |
| Koji su najvažniji rizici ulaganja?         | 0.11     | 0.63        | 0.49        | 0.79             |
| Koja je razlika između štednje i ulaganja?  | 0.02     | 0.31        | 0.31        | 0.76             |
| Kako mogu prepoznati financijsku prijevaru? | 0.01     | 0.23        | 0.17        | 0.70             |
| Što znači diverzifikacija portfelja?        | 0.02     | 0.26        | 0.17        | 0.79             |
| **Prosjek**                                 | **0.04** | **0.34**    | **0.28**    | **0.76**         |

---

**Tumačenje rezultata:**

* **BERTScore F1** je pokazao visoku semantičku sličnost između generiranih i idealnih odgovora (prosjek 0.76), što znači da je agent uspješno “pogodio bit” odgovora čak i kad su bile razlike u formulaciji.
* **ROUGE** i **BLEU** su očekivano niži jer strogo mjere preklapanje riječi, ali i dalje daju uvid u poklapanje ključnih termina.
* Ručna analiza potvrđuje da su odgovori većinom točni i korisni, uz povremene stilističke ili informacijske razlike.


| **Pitanje**                                        | **Odgovor chatbota**                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                         | **Ocjena** | **Komentar**                                                                                                                                                                             |
| -------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | ---------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Što je investicijski fond?                         | Investicijski fond je oblik zajednickog ulaganja kojim se skupljaju sredstva više ulagača kako bi se investirali u razne financijske instrumente.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            | 5          | Kratko, jasno, točno. Možda dodati primjer ili pojasniti “profesionalno upravljanje”.                                                                                                    |
| Koji su najvažniji rizici ulaganja?                | Neovisno o tome što uljevaš u ulaganja, najvažniji rizici su nepoznatost tržišta i nepoznatost kompanije u kojoj uljevaš. Zato je važno informirati se o svim troskovima i rizicima prije ulaganja i redovito pratiti svoja ulaganja kako bi se učio iz iskustva. Zapravo, iako ne ulazim novac koji mi je potreban za osnovne zivotne potrebe, i dalje bih se prethodno obavio istraživanja i analize tržišta i kompanije.                                                                                                                                                                                                                                                                                  | 3          | Previše osoban ton, navodi samo “nepoznatost” umjesto tipičnih rizika (tržišni, kreditni, likvidnosni, valutni…). Poželjno: nabrojati najčešće vrste rizika.                             |
| Koja je razlika između štednje i ulaganja?         | Štednja je stavljanje novca na račun ili u kasnu svrhu, obično u obliku sporih, bezobveznih obveza, kao što su sporozna stavka ili savings account. Ulaganje je stavljanje novca u neki tržišni instrument, kao što su akcije, obveznice ili fondovi za investiciju, s naadom da će se vratiti više novca u budućnosti. Štednja je tada konzervativnija opcija za financiranje dugoročnih ciljeva, kao što su štednja za kuću ili starost. Ulaganje je riskantnija opcija i može donijeti veće profitne rezultate ali i veći rizik za gubitak novca.                                                                                                                                                         | 4          | Jezgrena razlika dobro objašnjena, ali stil i jezik povremeno nespretan. “Kasnu svrhu” i “sporozna stavka” nisu tipične HR fraze. Dobro je što spominje rizik i profit.                  |
| Kako mogu započeti s ulaganjem u zlato?            | Prvi korak za započetak ulaganja u zlato je otvoriti račun u tvrtki koja se bavi prodajom i čuvanjem zlata, kao što je Centar Zlata. Na njemu možete otvoriti digitalni ili fizički način, na izbor imate i razne oblike ulaganja kao što su kupovina baračića ili kovanih kovčega zlata.                                                                                                                                                                                                                                                                                                                                                                                                                    | 4          | Točno, ali pomalo ograničeno na samo jedan primjer (“Centar Zlata”) i ne spominje druge načine (ETF, certifikati). Jezik: “baračići” nije standardan izraz.                              |
| Kako mogu prepoznati financijsku prijevaru?        | Ako ne poznavate investicijsku strategiju i ciljeve, te niste sposobni razumjeti finančne izvještaje i statistike, te nemate volju i vremena za redovito pratnje i prilagođavanje portfelja, tako ne možete u potpunosti prepoznati financijsku prijevaru. Ali, čak ni oni koji znaju, nisu potpuno sigurni od prijevare, jer financijski tržište je neprevidno i nepredvidivo. Jedino se može smanjiti rizik poštivanjem osnovnih principla tržišne vrijednosti i načela dobrih poslovanja.                                                                                                                                                                                                                 | 2          | Ne odgovara izravno na pitanje – govori o znanju općenito, ne o tipičnim znakovima prijevare (nerealna obećanja, neprovjereni izvori…). Treba biti konkretnije, više praktičnih savjeta. |
| Što znači diverzifikacija portfelja?               | Diversifikacija portfelja je strategija ulaganja koja se temelji na raspodjeli rizika prema što većem broju različitih klasa ulagačkih instrumenti, sa ciljem smanjenja rizika i povećanja povratka. Različitosti uključuju različite industrije, veličine kompanija i države ulaganja. Kada investitor ima raznoliko ulaganje, rizik od gubitka svega ulaganja kada se neka klasa gubi vrijednost smanjuje.                                                                                                                                                                                                                                                                                                 | 5          | Točno i jasno, dobra definicija, primjer naveden, sitne gramatičke greške ne umanjuju vrijednost odgovora.                                                                               |
| Kako se mogu informirati o ulaganjima u Hrvatskoj? | U Hrvatskoj postoje brojne izvori informacija koja mogu pomoći profesionalnim i kvalificiranim ulagateljima pri pripremi za ulaganje u fondove. Jedan od ključnih izvora informacija su prospekti fondova, koje su dostupni na web stranicama menadžera fondova ili Registarske agencije Hrvatske agencije za tržišne operacije (HAOT). Ostali izvori informacija uključuju novine, financijske časopisi i web stranice finansijskih institucija. Također, možete se odnositi na savet od strane finansijskih savetnika i menadžera fondova.                                                                                                                                                                 | 4          | Daje pravu smjernicu (prospekti fondova, savjetnici), ali nekoliko netočnosti: ne postoji HAOT, pravi naziv je HANFA. Stil je arhaičan (“odnositi na savet”), ali odgovor je solidan.    |
| Što su UCITS fondovi i koje su njihove prednosti?  | UCITS fondovi su profesionalno vođeni fondovi, koji se u skladu s propisima EU, moraju poštovati poznate standardne kriterije za razvoj, upravljanje i izvješćivanje. Njihove prednosti uključuju razumno visoke minimume ulaganja, transparenciju i lak pristup za investiciju, te kvalitetno obavljanje administrativnih poslova i osiguranje štednji. | 4          | Odgovor je uglavnom točan i jasno opisuje što su UCITS fondovi i navodi glavne prednosti (reguliranost, transparentnost, jednostavan pristup). Nedostaje malo više preciznosti (npr. što UCITS znači, prava definicija, diversifikacija, zaštita ulagatelja) i neki pojmovi (“razumno visoki minimumi ulaganja”) mogu biti zbunjujući – u praksi su minimumi često niski. Dobro bi bilo navesti i primjer ili dodatno pojasniti prednosti.                                                                           |
| Koje su prednosti i mane ulaganja u kriptovalute?  | Prednosti ulaganja u kriptovalute uključuju potencijalno visoke stope rasta vrijednosti, decentralizaciju i autonomiju, te visoku tehnološku inovaciju. Međutim, mane uključuju visoke rizike (poput tečajnih i tehničkih rizika), nezavršene tehnološke inovacije, kriminalne aktivnosti, i nepostojanje regulatorne okruženja. Uvijek se prije ulaganja informiraj o svim troskovima i rizicima, redovito prati svoja ulaganja i uci iz iskustva, te ne ulazi novac koji ti je potreban za osnovne zivotne potrebe.                                                                                                                                                                                        | 5          | Jasno navodi prednosti i mane, primjerena ravnoteža, samo nekoliko manjih gramatičkih grešaka.                                                                                           |
| Kako procijeniti isplativost nekog ulaganja?       | Kako biste procijenili isplativost nekog ulaganja, morate se usredotočiti na neke ključne faktorice: 1. Ciljani segment tržišta i konkurentsko okruženje. 2. Ekonomska stabilnost i politička situacija u državi gdje je fond smješten. 3. Stanje i trendovi tržišta i industrije koju pokriva fond. 4. Izraženost i iskustvo menadžera fonda i njegovih stručnjaka. 5. Učinak i isplativost fonda u prošlosti i trenutnom stanju. 6. Količina i struktura troškova (upravljačkih i drugih). 7. Ponajviše, obaviti što bolji analitički raspored i obaviti indepth istraživanje i analizu podataka koja su dostupna.                                                                                         | 4          | Dobro strukturiran odgovor, ali “faktorice” nije uobičajeno. Posljednja rečenica može biti jasnija, no informativno je.                                                                  |
| Koji su najčešći oblici financijske pismenosti?    | Financijska pismenost se može postići putem izučavanja tema kao što su osnovni finansijski izrazi i pojmovi, matematika vezana za financijske operacije, poznavanje financijskih instrumenta i tržišta, razumijevanje gospodarskih trendova i teorija financija.                                                                                                                                                                                                                                                                                                                                                                                                                                             | 4          | Dobro opisano, iako zapravo su “oblici” više “vještine” ili “područja”. Poželjno: konkretno nabrojati – “upravljanje budžetom”, “štednja”, “ulaganje”, “kredit”, “osiguranje”…           |
| Kako se zaštititi od lažnih investicijskih ponuda? | Kako bi se zaštitili od lažnih investicijskih ponuda, investitori trebaju provjereći povjerenika i fond, provjereći sadržaj ponude i tražiti savjete od stručnih savjetnika.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 | 4          | Kratko, ali dobro: moglo bi nabrojati još znakova prijevare (“previsoka zarada”, “pritisak na brzu odluku”).                                                                             |
| Što znači “passivno ulaganje”?                     | Passivno ulaganje znači investirati u fondove ili druge tržišne instrumente koji slijede performansu odabranog indeksa, a ne aktivan uložak. U tom slučaju, investitor ne može direktno utjecati na izbor kompanija ili odluke o ulaganju, već se oslanja na menadžere fonda ili indeks koji to rade za njih.                                                                                                                                                                                                                                                                                                                                                                                                | 5          | Točno i jasno, pohvala za opis “slijedi indeks”. Možda dodati primjere: ETF, indeksni fondovi.                                                                                           |
| Zašto je važno početi ulagati što ranije?          | Za početnike, važno je početi ulagati što ranije jer se vrijeme i rast kapitala prema zakonu trokuta utječu na konačni rezultat. Ulaganje ranije omogućava veću vremenu korištenje rastućem kapitalu i manje ovisnost o konkretnom trenutku. Zbog toga, što ranije počnete, to vam daje veći izbor strategija i raznolikijeg portfelja, te manje rizika od nepoželjnih izvjesenja.                                                                                                                                                                                                                                                                                                                           | 4          | Poanta je dobra (vrijeme i rast kapitala), ali “zakon trokuta” nije standardan izraz. Stil može biti jasniji.                                                                            |
| Koje su porezne obveze pri ulaganju u Hrvatskoj?   | Porezne obveze pri ulaganju u Hrvatskoj zavise od oblika ulaganja i vaše državljanstva. Obično se primjenjuju porez na kapitalni dobitak (10% za državljane i 15% za strance) i porez na dobitak od poslovanja (od 10 do 30%). Prilazite svom financijskom savjetniku ili Hrvatskom državnom agenciji za porez za točnije informacije.                                                                                                                                                                                                                                                                                                                                                                       | 3          | Djelomično točno, ali podaci nisu ažurni i mogu biti netočni (stope poreza). Dobro što upućuje na savjetnika/agenciju. Preporučuje se konkretan primjer i ažurirani podatak.             |



---

### **6. Opis dodatnih komponenti koje ste implementirali**

* **Streamlit GUI** s chatom i pregledom logova (JSON Lines)
* **Vizualizacija embeddinga** (t-SNE, Plotly)
* **Logiranje svake sesije u .jsonl datoteku** (pitanje, kontekst, odgovor, timestamp)
* **Pretraga i filtriranje po logovima**

---

### **7. Zaključak (refleksija, izazovi, prijedlozi)**

Projekt je pokazao da je moguće izraditi kvalitetnog lokalnog AI edukatora iz financija na relativno slabom računalu, koristeći open-source alate i vlastitu bazu znanja.
Najveći izazovi bili su optimizacija brzine odgovora (limit RAM-a i CPU-a), upravljanje tokenima u promptu te obrada PDF dokumenata različite kvalitete. Za poboljšanje: moguće je dodati naprednije izvore (tablice, slike), automatsku evaluaciju odgovora, višejezičnost i glasovni unos/izlaz.

---

### **8. Upute za pokretanje**

1. **Klonirati repozitorij**
2. Instalirati sve pakete:

   ```bash
   pip install streamlit langchain chromadb sentence-transformers llama-cpp-python pypdf scikit-learn plotly matplotlib nltk rouge-score bert-score
   ```
3. Preuzeti Mistral-7B-Instruct Q4\_K\_M .gguf model (npr. s [Hugging Face](https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.2-GGUF/blob/main/mistral-7b-instruct-v0.2.Q4_K_M.gguf)) i staviti u root projekta.
4. Pokrenuti aplikaciju:

   ```bash
   streamlit run main.py
   ```
5. Otvoriti browser na [http://localhost:8501](http://localhost:8501), koristiti chat i pregledavati logove.

### 9. Vizualizacija embeddinga

Vizualizacija je odrađena u Python Notebook datoteci _embedding_visualization.ipynb_ te se sa paketom _plotly_ izradi interaktivna mapa embeddinga.
