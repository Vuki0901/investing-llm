from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer

# Pitanja
pitanja = [
    "Što je investicijski fond?",
    "Koji su najvažniji rizici ulaganja?",
    "Koja je razlika između štednje i ulaganja?",
    "Kako mogu prepoznati financijsku prijevaru?",
    "Što znači diverzifikacija portfelja?"
]

# Očekivani (idealni) odgovori
ocekivani_odgovori = [
    "Investicijski fond je financijska institucija koja prikuplja sredstva više ulagatelja i ulaže ih u različite vrijednosne papire, čime omogućuje diverzifikaciju i profesionalno upravljanje ulaganjima.",
    "Najvažniji rizici ulaganja su tržišni, kreditni, likvidnosni i valutni rizik. Prije ulaganja važno je informirati se o svim mogućim rizicima.",
    "Štednja je odlaganje novca na sigurno mjesto bez rizika, dok ulaganje podrazumijeva preuzimanje rizika radi ostvarivanja većeg povrata.",
    "Financijsku prijevaru možete prepoznati po nerealnim obećanjima, neprovjerenim izvorima, pritisku na brzu odluku i izbjegavanju transparentnosti.",
    "Diversifikacija portfelja znači ulaganje u različite vrste imovine kako bi se smanjio ukupni rizik ulaganja."
]

# Odgovori chatbota (umetni svoje generirane odgovore)
odgovori_chatbota = [
    "Investicijski fond je oblik zajednickog ulaganja kojim se skupljaju sredstva više ulagača kako bi se investirali u razne financijske instrumente.",
    "Neovisno o tome što ulažeš u ulaganja, najvažniji rizici su nepoznatost tržišta i kompanije. Zato je važno informirati se o svim troškovima i rizicima prije ulaganja.",
    "Štednja je stavljanje novca na račun ili u kasnu svrhu, obično u obliku sporih, bezobveznih obveza. Ulaganje je stavljanje novca u neki tržišni instrument s nadom većeg povrata.",
    "Ako ne poznajete investicijsku strategiju i ciljeve te niste sposobni razumjeti financijske izvještaje i statistike, ne možete u potpunosti prepoznati financijsku prijevaru. Jedino se može smanjiti rizik poštivanjem osnovnih principa tržišne vrijednosti i dobrih poslovnih praksi.",
    "Diversifikacija portfelja je strategija ulaganja koja se temelji na raspodjeli rizika na što veći broj različitih klasa ulagačkih instrumenata radi smanjenja rizika i povećanja povrata."
]

# Inicijalizacija
scorer = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)
smooth = SmoothingFunction().method1

print("{:<60} {:<6} {:<8} {:<8}".format("Pitanje", "BLEU", "ROUGE-1", "ROUGE-L"))
print("-"*90)

for i, p in enumerate(pitanja):
    ref = ocekivani_odgovori[i]
    gen = odgovori_chatbota[i]
    bleu = sentence_bleu([ref.split()], gen.split(), smoothing_function=smooth)
    rouge = scorer.score(ref, gen)
    print("{:<60} {:.2f}   {:.2f}    {:.2f}".format(
        p, bleu, rouge['rouge1'].fmeasure, rouge['rougeL'].fmeasure
    ))