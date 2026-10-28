# Český hosting Pythonu do 2 000 Kč ročně

Ověřeno 28. 10. 2026. Ceny zahrnují DPH; úvodní bezplatná období nejsou započtena.

## Varianty

| Služba | Prostředky | Cena měsíčně s DPH | Přepočet ročně |
|---|---|---:|---:|
| Roští Small | 1 CPU, 2 GB RAM, 10 GB NVMe | Inzerováno 117 Kč za 30 dní | Přibližně 1 430 Kč za 365 dní |
| FORPSI Classic VPS Mini+ | 1 vCPU, 1 GB RAM, 20 GB SSD, IPv4/IPv6 | 84,70 Kč | 1 016,40 Kč |
| Web4ce VPS 1S | 1 vCPU, 1 GB RAM, 25 GB disku, IPv4/IPv6 | 108,90 Kč | 1 306,80 Kč |

Zdroje: [Roští ceník](https://rosti.cz/cenik), [FORPSI VPS](https://www.forpsi.com/virtual/), [Web4ce ceník](https://www.web4ce.cz/cz/virtualni-servery/cenik.html).

Roští účtuje po hodinách a měsíční cena je orientační pro 30 dní. Roční odhad vychází z 97 Kč bez DPH za 30 dní. Web4ce standardně vyžaduje objednávku minimálně na rok. FORPSI umožňuje měsíční, čtvrtletní i roční platbu. Ceny nezahrnují doplňky ani vlastní doménu.

## Roští: bez vlastní správy VPS

Pro nové projekty Roští doporučuje Hosting – Stacky, založený na Docker Compose. V jednom stacku může běžet Python aplikace i vlastní PostgreSQL/MariaDB a persistentní soubory. Dostupné jsou SSH, SFTP a automatické HTTPS. Zdroj: [dokumentace Stacků](https://docs.rosti.cz/cs/stacks/quickstart/), [Stacky versus Aplikace](https://docs.rosti.cz/cs/stacks/apps-vs-stacks/).

Důležité limity:

- Základní systém zabere přibližně 3 GB z diskové kvóty; do ní se počítá vše včetně systémových dat. Překročení se doúčtuje.
- Z internetu je dostupný jeden HTTP port přes reverzní proxy, ne libovolné veřejné TCP/UDP porty.
- Persistentní soubory mají být pod `/srv/stack`; preferované jsou bind mounty.
- Denní zálohy se uchovávají 14 dní. U vlastní databáze dokumentace doporučuje dělat také dumpy, protože filesystem snapshot nemusí zaručit konzistenci DB.
- Správa infrastruktury není totéž co správa vlastní aplikace a databázového kontejneru.

Zdroj: [Quickstart Stacků](https://docs.rosti.cz/cs/stacks/quickstart/).

Ceník také uvádí sdílenou databázi Free s 1 GB a limitem 250 000 dotazů za hodinu. Návrh výše s ní ale nepočítá: databáze běží přímo ve stacku. Vlastní `.cz` doména stojí 350,90 Kč ročně; Small s touto doménou tedy vychází přibližně na 1 780 Kč za rok. Zdroj: [ceník](https://rosti.cz/cenik).

## VPS alternativy

FORPSI Mini+ je levnější varianta pro vlastní správu, ale má pouze 1 GB RAM a tarif nelze přímo upgradovat. Web4ce nabízí 25 GB disku a standardně roční objednávku. U obou je potřeba zvlášť ověřit rozsah a cenu záloh a případné správy; vlastní aplikaci a DB nasazuje zákazník. Zdroje: [FORPSI](https://www.forpsi.com/virtual/), [Web4ce](https://www.web4ce.cz/cz/virtualni-servery/cenik.html).

## České společnosti

- **Roští.cz, s.r.o.**, IČO 04173571, Veverkova 1343/1, Hradec Králové. [Smluvní podmínky](https://rosti.cz/en/smluvni-podminky).
- **INTERNET CZ, a.s. (FORPSI)**, IČO 26043319, Ktiš 2, 384 03 Ktiš. [Kontakty](https://www.forpsi.com/contacts/?lang=en-us).
- **Web4ce, s.r.o.**, IČO 27148122, Pšenčíkova 674, Praha 4. [Obchodní podmínky](https://www.web4ce.cz/cz/spolecnost/obchodni-podminky.html).

Před zaplacením zkontrolujte, že tato společnost skutečně vystupuje jako poskytovatel ve smlouvě a dodavatel na faktuře.

## DPH: podstatné rozlišení

Pro podnikatele-neplátce může přijetí služby s místem plnění v ČR od osoby neusazené v tuzemsku založit status identifikované osoby podle § 6h. Nejde o automatický přechod do běžného plátcovství DPH. Povinnost může vzniknout již poskytnutím zálohy. Zdroje: [§ 6h](https://krajta.slv.cz/2004/235/par_6h), [informace Finanční správy ke změnám od roku 2025](https://financnisprava.gov.cz/assets/cs/prilohy/d-seznam-dani/82973_Souhrnna-informace-k-novele-DPH-2025.pdf).

Běžný tuzemský nákup od českého dodavatele tento konkrétní zahraniční registrační důvod nezakládá. Česká doména, lokalita serveru ani česká podpora samy o sobě nestačí; podstatný je skutečný poskytovatel.

U služby pořízené výlučně pro soukromou potřebu se příjemce pro pravidla místa plnění považuje za osobu nepovinnou k dani. Samotný soukromý nákup zahraničního hostingu tedy není stejná situace jako podnikatelský nákup podle § 6h. Musí ale skutečně jít o soukromé využití, nikoli jen o vynechání IČO. Zdroj: [článek 19 prováděcího nařízení EU 282/2011](https://eur-lex.europa.eu/legal-content/CS/TXT/HTML/?uri=CELEX:02011R0282-20250414).

Jde o obecné vysvětlení, nikoli individuální daňové stanovisko.
