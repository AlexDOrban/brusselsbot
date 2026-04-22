"""Build data/benchmark_questions.json (50 questions across 6 categories).

Uses chunks.json to resolve (lang, article) or (source, page) → chunk_id so
expected_sources lists are accurate after chunk regeneration is idempotent.
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CHUNKS = json.loads((ROOT / "data" / "chunks.json").read_text())

# Lookups
ART = {}  # (lang, article_number) -> chunk_id
SRC_PAGE = {}  # (source, page) -> list[chunk_id]
for i, c in enumerate(CHUNKS):
    md = c["metadata"]
    cid = f"chunk_{i}"
    if md.get("source") == "police_regs" and md.get("article_number"):
        ART[(md["language"], md["article_number"])] = cid
    if md.get("source") in ("ieep", "leaseplan"):
        SRC_PAGE.setdefault((md["source"], md.get("page")), []).append(cid)


def fr(art):
    return ART[("fr", art)]


def nl(art):
    return ART[("nl", art)]


def ieep(page):
    return SRC_PAGE[("ieep", page)][0]


def lp(page):
    return SRC_PAGE[("leaseplan", page)][0]


def ieep_any():
    return [cid for (s, _), cids in SRC_PAGE.items() if s == "ieep" for cid in cids]


def lp_any():
    return [cid for (s, _), cids in SRC_PAGE.items() if s == "leaseplan" for cid in cids]


Q = []


def add(qid, category, question, language, expected_sources, notes=""):
    Q.append({
        "id": qid,
        "category": category,
        "question": question,
        "language": language,
        "expected_sources": expected_sources,
        "ground_truth_answer": "draft — manual review required",
        "notes": notes,
    })


# ── in_scope_fr (10) ──────────────────────────────────────────────────────────
add("Q6_grue_fr", "in_scope_fr",
    "Quelles sont les règles pour l'installation d'une grue?",
    "fr", [fr(54)],
    "preserved from original 10")
add("Q11_alarme_fr", "in_scope_fr",
    "Quelles sont les obligations pour les systèmes d'alarme sonores à Bruxelles?",
    "fr", [fr(97), fr(96)])
add("Q12_trottoir_fr", "in_scope_fr",
    "Qui est responsable du nettoyage du trottoir devant mon immeuble?",
    "fr", [fr(21), fr(22)])
add("Q13_feu_fr", "in_scope_fr",
    "Est-il permis d'allumer un feu ouvert dans son jardin?",
    "fr", [fr(81), fr(47)])
add("Q14_chantier_fr", "in_scope_fr",
    "Quelles autorisations sont nécessaires pour un chantier sur la voie publique?",
    "fr", [fr(55), fr(20)])
add("Q15_demenage_fr", "in_scope_fr",
    "Quelles sont les règles pour un déménagement à Bruxelles?",
    "fr", [fr(63)])
add("Q16_chien_fr", "in_scope_fr",
    "Que dois-je faire des excréments de mon chien dans l'espace public?",
    "fr", [fr(115)])
add("Q17_deposantclandestin_fr", "in_scope_fr",
    "Quelle sanction pour un dépôt clandestin de déchets?",
    "fr", [fr(25), fr(14), fr(4)])
add("Q18_affichage_fr", "in_scope_fr",
    "Puis-je coller des affiches sur un mur public?",
    "fr", [fr(38), fr(39), fr(40)])
add("Q19_taxi_fr", "in_scope_fr",
    "Quelles règles s'appliquent aux voitures immobilisées sur la voie publique?",
    "fr", [fr(47)])

# ── in_scope_nl (10) ──────────────────────────────────────────────────────────
add("Q2_muziek_nl", "in_scope_nl",
    "Mag ik 's nachts muziek maken op straat?",
    "nl", [nl(88)],
    "preserved")
add("Q3_sluikstort_nl", "in_scope_nl",
    "Welke boete riskeer ik voor sluikstort?",
    "nl", [nl(25), nl(14), nl(4), nl(29)],
    "preserved; BM25-critical")
add("Q20_alarm_nl", "in_scope_nl",
    "Wat zijn de regels voor alarmsystemen op gebouwen?",
    "nl", [nl(97), nl(96)])
add("Q21_verhuizing_nl", "in_scope_nl",
    "Welke toestemming heb ik nodig voor een verhuizing?",
    "nl", [nl(63)])
add("Q22_voetpad_nl", "in_scope_nl",
    "Wie moet het voetpad voor mijn woning sneeuwvrij houden?",
    "nl", [nl(21), nl(22)])
add("Q23_werf_nl", "in_scope_nl",
    "Welke regels gelden voor een bouwwerf op de openbare weg?",
    "nl", [nl(54), nl(17)])
add("Q24_hond_nl", "in_scope_nl",
    "Mag mijn hond loslopen in een park in Brussel?",
    "nl", [nl(112)])
add("Q25_afvalzak_nl", "in_scope_nl",
    "Wanneer mag ik mijn huisvuilzak buiten zetten?",
    "nl", [nl(28), nl(26)])
add("Q26_gezondheid_nl", "in_scope_nl",
    "Kan de gemeente een ongezonde woning sluiten?",
    "nl", [nl(24), nl(30)])
add("Q27_vuur_nl", "in_scope_nl",
    "Mag ik bladeren verbranden in mijn tuin?",
    "nl", [nl(43)])

# ── in_scope_en (10) ──────────────────────────────────────────────────────────
add("Q1_noise_en", "in_scope_en",
    "What are the rules about noise at night in Brussels?",
    "en", [fr(88), nl(88)],
    "preserved")
add("Q4_lez_social", "in_scope_en",
    "How has the Brussels LEZ affected low-income households?",
    "en", ieep_any(),
    "preserved; any IEEP chunk = hit")
add("Q5_bruxellair", "in_scope_en",
    "What is the Bruxell'Air bonus?",
    "en", [ieep(6), ieep(7)],
    "preserved")
add("Q8_ev_fleet", "in_scope_en",
    "What electric vehicle incentives are available for business fleets?",
    "en", lp_any(),
    "preserved")
add("Q10_alarm", "in_scope_en",
    "Can my neighbour's alarm ring all night?",
    "en", [fr(97), nl(97)],
    "preserved")
add("Q7_trash_sidewalk", "in_scope_en",
    "Can I put a trash container on the sidewalk?",
    "en", [fr(28), nl(28), fr(29), fr(30)],
    "preserved; EN query, FR/NL source — kept in_scope_en per task guidance")
add("Q28_crane_en", "in_scope_en",
    "What permits do I need to install a crane on a construction site in Brussels?",
    "en", [fr(54), nl(54)])
add("Q29_fireworks_en", "in_scope_en",
    "Are fireworks allowed in Brussels?",
    "en", [fr(43), nl(43)])
add("Q30_lez_enforcement", "in_scope_en",
    "How is the Low Emission Zone enforced in Brussels?",
    "en", ieep_any())
add("Q31_lp_modalshift", "in_scope_en",
    "What does the LeasePlan whitepaper say about modal shift in Belgium?",
    "en", lp_any())

# ── out_of_scope (5) ──────────────────────────────────────────────────────────
add("Q9_e40_oos", "out_of_scope",
    "What is the fine for speeding on the E40 highway?",
    "en", [],
    "preserved; federal traffic law")
add("Q32_tax_oos", "out_of_scope",
    "What is the personal income tax rate for Brussels residents?",
    "en", [],
    "regional fiscal law, not in corpus")
add("Q33_parking_oos", "out_of_scope",
    "How much does a Brussels resident parking permit cost per year?",
    "en", [],
    "municipal fees schedule, not in police regs")
add("Q34_eu_emissions_oos", "out_of_scope",
    "What are the EU Euro 7 emission standards?",
    "en", [],
    "EU-level regulation, not in corpus")
add("Q35_customs_oos", "out_of_scope",
    "What documents do I need to import a car from Germany to Belgium?",
    "en", [],
    "federal customs, not in corpus")

# ── multi_hop (10) — require 2+ distinct articles ─────────────────────────────
add("Q36_noise_and_construction_fr", "multi_hop",
    "Puis-je faire des travaux bruyants dans mon jardin un dimanche matin?",
    "fr", [fr(88), fr(91), fr(54)],
    "noise rules + construction hours + site rules")
add("Q37_dog_and_dumping_fr", "multi_hop",
    "Quelle est la différence entre abandonner un déchet et laisser des excréments de chien?",
    "fr", [fr(25), fr(115), fr(4)])
add("Q38_party_noise_sanction_fr", "multi_hop",
    "Si j'organise une fête bruyante, quelle sanction je risque et à quelle heure dois-je arrêter?",
    "fr", [fr(88), fr(4), fr(91)])
add("Q39_cleaning_snow_nl", "multi_hop",
    "Wie moet het voetpad sneeuwvrij maken en wat is de sanctie bij niet-naleving?",
    "nl", [nl(21), nl(22), nl(4)])
add("Q40_alarm_and_noise_nl", "multi_hop",
    "Wat zijn de regels voor alarmsystemen, en gelden de nachtelijke geluidsregels ook voor alarmen?",
    "nl", [nl(97), nl(88), nl(96)])
add("Q41_waste_and_moving_nl", "multi_hop",
    "Bij een verhuizing: waar mag ik het afval laten staan en welke toestemming heb ik nodig?",
    "nl", [nl(63), nl(28), nl(14)])
add("Q42_crane_and_sidewalk_en", "multi_hop",
    "If I set up a crane on the sidewalk for a move, which rules do I need to follow?",
    "en", [fr(54), fr(63), fr(21), nl(54), nl(63), nl(21)])
add("Q43_sanctions_lez_en", "multi_hop",
    "What sanctions apply if I drive a non-compliant vehicle in the Brussels LEZ, and how does this compare to police regulation fines?",
    "en", ieep_any() + [fr(4)])
add("Q44_fire_and_smoke_fr", "multi_hop",
    "Puis-je faire un barbecue sur mon balcon, et quelles sont les règles sur la fumée?",
    "fr", [fr(43), fr(81)])
add("Q45_dog_park_sanction_nl", "multi_hop",
    "Mag mijn hond loslopen in het park, en wat riskeer ik als hij zijn behoefte doet zonder dat ik opruim?",
    "nl", [nl(112), nl(115), nl(4)])

# ── cross_lingual (5) — question in one language, sources in another ──────────
add("Q46_fr_to_nl", "cross_lingual",
    "Quelles sont les règles bruxelloises sur le bruit nocturne? (répondre en français mais les sources sont en néerlandais)",
    "fr", [nl(88), nl(91)],
    "FR query, NL expected_sources — forces cross-lingual retrieval")
add("Q47_nl_to_fr", "cross_lingual",
    "Welke Brusselse regels gelden voor straatverkopen?",
    "nl", [fr(38), fr(39), fr(40)],
    "NL query, FR sources on sales/display")
add("Q48_en_to_fr", "cross_lingual",
    "What are the Brussels rules for installing scaffolding on a construction site?",
    "en", [fr(54), fr(55)],
    "EN query, FR-only sources")
add("Q49_en_to_nl", "cross_lingual",
    "Are there Brussels rules about holding a public demonstration?",
    "en", [nl(11), nl(12)],
    "EN query, NL sources on assembly")
add("Q50_fr_to_en_ieep", "cross_lingual",
    "Comment la zone de basses émissions a-t-elle affecté les ménages modestes à Bruxelles?",
    "fr", ieep_any(),
    "FR query, EN-only IEEP sources")


def main():
    out = {"questions": Q}
    path = ROOT / "data" / "benchmark_questions.json"
    path.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    # sanity
    from collections import Counter
    cats = Counter(q["category"] for q in Q)
    print(f"Saved {len(Q)} questions → {path}")
    print("By category:", dict(cats))
    assert len(Q) == 50, f"expected 50, got {len(Q)}"
    assert cats == {"in_scope_fr": 10, "in_scope_nl": 10, "in_scope_en": 10,
                    "out_of_scope": 5, "multi_hop": 10, "cross_lingual": 5}, cats
    # assert all expected_sources entries are real chunk IDs (except OOS)
    valid = {f"chunk_{i}" for i in range(len(CHUNKS))}
    for q in Q:
        for cid in q["expected_sources"]:
            assert cid in valid, f"{q['id']}: unknown chunk_id {cid}"
    print("All chunk_ids valid.")


if __name__ == "__main__":
    main()
