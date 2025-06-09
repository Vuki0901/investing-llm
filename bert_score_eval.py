from bert_score import score

# Očekivani (idealni) odgovori
ocekivani_odgovori = [
    "Investicijski fond je financijska institucija koja prikuplja sredstva više ulagatelja i ulaže ih u različite vrijednosne papire, čime omogućuje diverzifikaciju i profesionalno upravljanje ulaganjima.",
    "Najvažniji rizici ulaganja su tržišni, kreditni, likvidnosni i valutni rizik. Prije ulaganja važno je informirati se o svim mogućim rizicima.",
    "Štednja je odlaganje novca na sigurno mjesto bez rizika, dok ulaganje podrazumijeva preuzimanje rizika radi ostvarivanja većeg povrata.",
    "Financijsku prijevaru možete prepoznati po nerealnim obećanjima, neprovjerenim izvorima, pritisku na brzu odluku i izbjegavanju transparentnosti.",
    "Diversifikacija portfelja znači ulaganje u različite vrste imovine kako bi se smanjio ukupni rizik ulaganja."
]

# Odgovori chatbota
odgovori_chatbota = [
    "Investicijski fond je oblik zajednickog ulaganja kojim se skupljaju sredstva više ulagača kako bi se investirali u razne financijske instrumente.",
    "Neovisno o tome što ulažeš u ulaganja, najvažniji rizici su nepoznatost tržišta i kompanije. Zato je važno informirati se o svim troškovima i rizicima prije ulaganja.",
    "Štednja je stavljanje novca na račun ili u kasnu svrhu, obično u obliku sporih, bezobveznih obveza. Ulaganje je stavljanje novca u neki tržišni instrument s nadom većeg povrata.",
    "Ako ne poznajete investicijsku strategiju i ciljeve te niste sposobni razumjeti financijske izvještaje i statistike, ne možete u potpunosti prepoznati financijsku prijevaru. Jedino se može smanjiti rizik poštivanjem osnovnih principa tržišne vrijednosti i dobrih poslovnih praksi.",
    "Diversifikacija portfelja je strategija ulaganja koja se temelji na raspodjeli rizika na što veći broj različitih klasa ulagačkih instrumenata radi smanjenja rizika i povećanja povrata."
]

# BERTScore izračun (za hrvatski koristi lang="hr", ali podržani su i "en", "de", "fr", ...; ako "hr" ne radi, koristi "multilingual")
P, R, F1 = score(odgovori_chatbota, ocekivani_odgovori, lang="hr")

print("{:<60} {:<8}".format("Pitanje", "BERTScore F1"))
print("-"*75)
for i, f1 in enumerate(F1):
    print("{:<60} {:.2f}".format(f"Pitanje {i+1}", f1))
print("-"*75)
print("Prosječni BERTScore F1:", round(float(F1.mean()), 2))
