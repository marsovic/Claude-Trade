# L'app mobile du bot (Google AI Studio)

Objectif : voir sur ton téléphone ce que montre le tableau de bord de ton Mac (capital,
positions, candidats du jour, journal, trades), où que tu sois.

## Est-ce faisable ? Oui, avec une condition

Le bot et son tableau de bord tournent sur ton Mac. Le téléphone doit donc pouvoir
**joindre ton Mac de façon sûre**. La bonne solution est **Tailscale**, gratuit en usage
perso. C'est un réseau privé entre tes appareils : le téléphone voit le Mac comme s'il
était sur le même Wi-Fi, même en 4G, et personne d'autre n'y a accès.

**N'ouvre jamais le port du tableau de bord sur Internet** : pas de redirection sur la
box, pas de « Tailscale Funnel ».

L'architecture :

```
Mac : bot -> state_insider_trading212/ -> tableau de bord (127.0.0.1:8051)
                                               |  tailscale serve (HTTPS privé)
Téléphone (Tailscale actif) -> https://ton-mac.xxxx.ts.net  -> /api/state (JSON)
   a) le tableau de bord tel quel, dans le navigateur      (aucun développement)
   b) l'app générée par Google AI Studio, qui lit /api/state (lecture seule)
```

**L'app ne peut ni passer ni annuler d'ordre** : l'API est en lecture seule, et c'est voulu.

## Étape 1 — Tailscale (15 minutes)

1. Installe Tailscale sur le Mac (tailscale.com/download) et sur le téléphone (App Store
   ou Play Store). Connecte les deux avec le **même compte**.
2. Sur login.tailscale.com, page DNS : active **MagicDNS**, puis **HTTPS Certificates**.
3. Sur le Mac, dans le Terminal :
   ```bash
   tailscale serve --bg 8051
   tailscale serve status        # affiche l'adresse https://<ton-mac>.<xxxx>.ts.net
   ```
   (avec l'app Mac, la commande est
   `/Applications/Tailscale.app/Contents/MacOS/Tailscale serve --bg 8051`)

## Étape 2 — Protéger l'API par un mot de passe

Dans le `.env` du projet, ajoute :

```bash
DASHBOARD_TOKEN=<une longue chaîne aléatoire>
DASHBOARD_CORS_ORIGIN=*
```

Pour générer la chaîne : `python3 -c "import secrets; print(secrets.token_urlsafe(24))"`.

`*` autorise l'app AI Studio à lire l'API. Ce n'est sans risque que parce que le token
reste obligatoire et que seuls tes appareils Tailscale atteignent le Mac. Une fois l'app
publiée, tu peux remplacer `*` par son adresse exacte (par exemple
`https://mon-app-xxxx.run.app`).

Redémarre le tableau de bord :

```bash
launchctl kickstart -k gui/$(id -u)/com.claudetrade.dashboard
```

## Étape 3 — Option immédiate : le tableau de bord sur le téléphone

Sur le téléphone, avec Tailscale actif, ouvre
`https://<ton-mac>.<xxxx>.ts.net/#token=<ton token>`. Ajoute ensuite la page à l'écran
d'accueil :

- **iPhone** (Safari) : Partager, puis « Sur l'écran d'accueil » ;
- **Android** (Chrome) : menu ⋮, puis « Ajouter à l'écran d'accueil ».

Tu as déjà tout, sans rien développer.

## Étape 4 — L'app sur mesure avec Google AI Studio

1. Va sur aistudio.google.com, puis **Build** (création d'app).
2. Colle le prompt ci-dessous en entier.
3. Teste dans l'aperçu. Dans les Réglages de l'app, saisis l'adresse
   `https://<ton-mac>.<xxxx>.ts.net` et ton token. Le Mac doit être sur Tailscale.
4. **Deploy** (Cloud Run) pour obtenir une adresse fixe. Ouvre-la sur le téléphone et
   ajoute-la à l'écran d'accueil comme à l'étape 3.

Bon à savoir :

- L'app tourne dans le navigateur du téléphone : Tailscale doit être actif pour qu'elle
  joigne le Mac.
- Le résumé par Gemini envoie le journal du jour à Google. Pour un usage perso, c'est
  acceptable, et il n'y a ni clé ni identifiant dedans.

---

## Le prompt à coller dans AI Studio

````text
Crée une application web mobile (React + TypeScript, mobile-first, utilisable en PWA « ajouter à l'écran d'accueil ») en FRANÇAIS, pour un usage personnel : suivre un robot de trading d'actions US qui tourne sur mon ordinateur. L'app est STRICTEMENT EN LECTURE SEULE : elle ne doit jamais proposer de passer, modifier ou annuler un ordre.

## Le robot (contexte)
- Il démarre avec 100 € sur un compte démo Trading 212. S'il descend à 10 €, il s'arrête définitivement (« règle de survie »).
- Stratégie « insider » : chaque nuit, il lit les déclarations SEC Form 4 de la veille, c'est-à-dire les achats d'actions par les dirigeants. Il garde les achats sur le marché du PDG ou du directeur financier (≥ 50 000 $), les gros achats (≥ 100 000 $ et +20 % de sa ligne) et les achats de dirigeants ou d'administrateurs (≥ 25 000 $). Il vérifie chaque candidat : prix ≥ 2 $, ≥ 1 M$ échangés par jour, prix déclaré cohérent avec le marché, pas d'annonce de contrat ou de financement (8-K 1.01) ni de chute après résultats dans les 20 dernières séances, action disponible sur Trading 212, place libre (4 positions maximum).
- Il achète à l'ouverture suivante (15 h 30, heure de Paris), avec un stop de sécurité à -35 %, et revend après 20 séances.

## Source des données
- Une seule requête : GET {baseUrl}/api/state avec l'en-tête « Authorization: Bearer {token} ».
- baseUrl (ex. https://mon-mac.tailxxxx.ts.net) et token se saisissent dans un écran Réglages et sont gardés en localStorage. Le token n'est jamais affiché en clair après la saisie.
- Rafraîchissement toutes les 30 secondes au premier plan, plus un « tirer pour rafraîchir ».
- Erreurs à afficher clairement : 401 « token incorrect » ; réseau « Mac injoignable : Tailscale est-il actif sur le téléphone, et le Mac allumé ? ». Garder la dernière réponse reçue avec l'heure « mis à jour il y a X min ».
- Prévoir un « mode démo » avec des données fictives au même format, pour tester l'app sans le Mac.

## Format de la réponse JSON (champs utiles)
{
  "now": 1760000000000,                  // ms, heure du Mac
  "tombstone": null | {"died_at_iso": "...", "equity": 9.5, "trades": 12},   // non nul = robot mort
  "state": {
    "status": "running" | "stopped" | "dead",
    "mode": "trading212-paper",
    "strategy": "insider",
    "initial_capital": 100.0,
    "equity": 103.2,                     // capital actuel en EUR
    "peak_equity": 105.0,
    "cash": 40.1,
    "realized_eur": 3.4, "fees_eur": 0.6,
    "daily_halt": false,                 // true = nouvelles entrées en pause
    "pause_until": "",
    "last_day": "2026-10-16",            // dernière séance traitée
    "insider_scanned": "2026-10-16",
    "heartbeat": 1760000000000,          // ms : dernier signe de vie du robot
    "next_scan_at": 1760100000000,       // ms : prochaine analyse prévue
    "equity_history": [[ts_ms, equity_eur], ...],
    "stats": {"trades": 5, "wins": 3, "losses": 2, "gross_profit": 4.1, "gross_loss": 1.2, "fees": 0.5},
    "positions": {
      "BYRN": {"symbol": "BYRN", "qty": 4.09, "entry_price": 4.69, "last_price": 4.95,
               "cost_eur": 17.6, "unrealized_eur": 0.9, "stop_loss": 3.05,
               "entry_day": "2026-10-20", "currency": "USD", "broker_ticker": "BYRN_US_EQ"}
    },
    "pending": [ {"symbol": "PAM", "side": "BUY", "qty": 0.346, "kind": "LIMIT",
                  "limit_price": 87.2, "reason": "entry", "day": "2026-10-19"} ],
    "last_scan": {
      "day": "2026-10-16",
      "conditions": ["rule","price","liquidity","price_check","no_veto","tradable","slot"],
      "condition_labels": ["insider rule","price >= $2","volume >= $1M/day","insider price ~ market","no 8-K 1.01 / earnings drop","tradable","free slot"],
      "evaluations": [
        {"symbol": "BYRN", "signal": "LONG" | null, "price": 4.69,
         "long": {"rule": true, "price": true, "liquidity": true, "price_check": true,
                  "no_veto": true, "tradable": true, "slot": true},
         "long_score": 7, "values": {"value_k": 70.4, "dv_musd": 3.2}}
      ],
      "signals": ["BYRN"]
    },
    "settings": {"insider_slots": 4, "insider_hold_days": 20, "insider_stop_pct": 0.35,
                 "survival_floor": 10.0}
  },
  "decisions": [   // journal, le plus récent en premier
    {"ts": 1760000000000, "kind": "order" | "skip" | "open" | "close" | "stop" | "warning" | "error" | "info" | "halt" | "death" | "start",
     "symbol": "BYRN" | null, "side": "LONG" | null, "message": "texte en anglais"}
  ],
  "trades": [      // trades clôturés, le plus récent en premier
    {"symbol": "PAM", "qty": 0.3, "entry_price": 80.8, "exit_price": 85.1, "pnl": 1.12,
     "r_multiple": 0.15, "reason": "time exit (20 sessions held)", "opened_at": ms, "closed_at": ms}
  ]
}
Les champs inconnus doivent être ignorés sans planter. Les nombres peuvent être null.

## Écrans (barre d'onglets en bas)
1. **Accueil**
   - Grande carte du capital en € avec la variation depuis 100 € (en % et en €), et la distance à la règle de survie (capital - 10 €).
   - Badge d'état : « Actif » (vert) ; « En pause » si daily_halt ; « Arrêté » ; « MORT » (rouge) si tombstone. Badge « Robot muet » (orange) si heartbeat date de plus de 10 minutes un jour de semaine.
   - Courbe du capital (equity_history), avec choix 1 sem. / 1 mois / tout.
   - Ligne « Prochaine analyse : … » (next_scan_at, heure de Paris) et « Dernière séance traitée : … ».
   - Stats : nombre de trades, taux de réussite, P&L réalisé, frais.
2. **Positions** : une carte par position avec :
   - le symbole ;
   - la quantité ;
   - le prix d'entrée et le dernier prix ;
   - le P&L latent en € et en % ;
   - le stop ;
   - le jour d'entrée, avec « séance X / 20 » estimée en jours ouvrés depuis entry_day ;
   - une barre de progression vers la sortie.

   Afficher aussi les ordres en attente (pending) : « Achat prévu à l'ouverture ».
3. **Candidats** : les evaluations de last_scan, une ligne par action.
   - Une pastille par condition (verte ou grise, libellés français) et un badge « Acheté » si signal.
   - Le montant de l'achat d'initié (values.value_k en milliers de $).
   - Un tap ouvre le détail avec le message du journal correspondant (decisions avec le même symbole, kind order ou skip).
4. **Journal** : liste des decisions avec filtres Tout / Ordres / Refus (skip) / Alertes (warning, error, halt, death).
   - Heure locale de Paris, symbole en gras, message.
   - Couleurs : rouge pour error et death, orange pour warning et halt.
5. **Trades** : liste des trades clôturés (P&L en couleur, raison, durée), avec le total en haut.
6. **Réglages** : baseUrl, token, mode démo, bouton « Tester la connexion ».

## Fonction Gemini (optionnelle)
Un bouton « Résumé du jour » sur l'Accueil. Il envoie à Gemini les positions, les candidats et les 50 dernières entrées du journal, et demande un résumé en 5 phrases simples en français. Ce résumé dit ce que le robot a acheté, refusé et pourquoi, et s'il y a une alerte. Il ne donne JAMAIS de conseil d'achat ou de vente, et indique que c'est un compte démo.

## Design
- Thème sombre par défaut et thème clair automatique, gros chiffres lisibles.
- Montants en € avec 2 décimales, prix en $.
- Toutes les heures en heure de Paris (Europe/Paris).
- Aucune donnée envoyée ailleurs que vers baseUrl, sauf la fonction Gemini sur demande.
````
