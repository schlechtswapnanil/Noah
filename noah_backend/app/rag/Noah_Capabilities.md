# About Noah

User-facing source for questions about Noah itself: what it is, what it can do,
where its data comes from, and what it cannot do. Noah quotes this text, so keep
it accurate — if behaviour changes, change this file. Each `##` section is
retrieved as a unit, and each is written in English and then German so that a
question asked in either language matches.

## What Noah is and what it can do

Noah is the AI assistant built into the PayTo shopping app. It turns a request
written in plain language into an action inside the app. Noah can search for
products, offers, deals and discounts, compare prices, check whether an item is
in stock, open loyalty cards and barcodes from your wallet, show your points and
rewards, find and navigate to nearby stores, check opening hours, build a
shopping list, and work out the cheapest way to buy a whole basket of groceries
across several supermarkets.

Noah ist der KI-Assistent in der PayTo-App. Noah kann Produkte, Angebote und
Rabatte suchen, Preise vergleichen, die Verfügbarkeit prüfen, Treuekarten und
Barcodes aus deiner Wallet öffnen, Punkte und Prämien anzeigen, Märkte in der
Nähe finden und dorthin navigieren, Öffnungszeiten nennen, Einkaufslisten
erstellen und den günstigsten Einkaufskorb über mehrere Supermärkte hinweg
berechnen.

## Real-time price data and where prices come from

Prices, offers and stock are real-time live data, not stored or cached numbers.
When you ask, Noah queries the OfferHopper service at that moment and returns
what the stores are offering right now. Coverage is German supermarkets and
drugstores including Aldi, Lidl, REWE, Edeka, Penny, Netto, Kaufland, Globus,
dm, Rossmann and Müller. If that live service cannot be reached, Noah says the
price service is unavailable rather than showing an old or estimated price.

Die Preise, Angebote und Verfügbarkeiten sind Echtzeitdaten und werden live
abgefragt, nicht gespeichert. Noah fragt im Moment deiner Anfrage den Dienst
OfferHopper ab und zeigt, was die Märkte gerade anbieten. Abgedeckt sind
deutsche Supermärkte und Drogerien wie Aldi, Lidl, REWE, Edeka, Penny, Netto,
Kaufland, Globus, dm, Rossmann und Müller. Ist der Dienst nicht erreichbar,
sagt Noah das, statt einen veralteten oder geschätzten Preis zu nennen.

## RAG, retrieval and how questions about PayTo are answered

Yes, Noah uses RAG — retrieval-augmented generation. For questions about PayTo
itself, such as privacy, loyalty points or how a feature works, Noah searches
PayTo's published documentation and FAQ, retrieves the passages that match, and
answers from that retrieved text rather than from a language model's memory.
This is what keeps answers grounded and stops the assistant inventing facts. If
no relevant passage is found, Noah says it does not have that information
instead of guessing. Only documents written for users are searchable; internal
engineering documents are not.

Ja, Noah nutzt RAG, also retrieval-augmented generation. Bei Fragen zu PayTo
selbst durchsucht Noah die veröffentlichte Dokumentation und die FAQ, holt die
passenden Abschnitte und antwortet auf Basis dieses Textes statt aus dem
Gedächtnis eines Sprachmodells. Findet Noah nichts Passendes, sagt Noah das,
statt zu raten.

## AI, machine learning and how Noah understands a request

Noah uses machine learning to understand you. Every request is classified by a
trained model into an intent and a plan of app actions, and the model returns a
confidence score with its prediction. Below a set confidence threshold Noah does
not act: it asks a clarifying question instead. Acting on a misread request
means opening the wrong screen or spending a live price lookup, so an uncertain
request is worth one extra question. A language model is used only to phrase the
final reply from facts Noah already has — never to decide what action to take.

Noah nutzt maschinelles Lernen und künstliche Intelligenz. Jede Anfrage wird von
einem trainierten Modell einer Absicht und einem Handlungsplan zugeordnet, mit
einem Konfidenzwert. Liegt dieser unter einem Schwellenwert, handelt Noah nicht,
sondern stellt eine Rückfrage. Ein Sprachmodell formuliert nur die Antwort, es
entscheidet nie über die Aktion.

## Languages Noah speaks

Noah understands and replies in English and German, including informal phrasing,
regional wording and typos. Noah answers in whichever of the two you wrote in.

Noah versteht und antwortet auf Englisch und Deutsch, auch bei umgangssprachlichen
Formulierungen und Tippfehlern, und antwortet in der Sprache deiner Frage.

## Memory and conversation history

Noah does not remember earlier messages and keeps no conversation history. Each
request is handled on its own, so a follow-up such as "and then navigate there"
or "the cheapest one" has no previous turn to refer to, and Noah will ask you to
say it in full.

Noah merkt sich frühere Nachrichten nicht und speichert keinen Verlauf. Jede
Anfrage steht für sich. Bei Rückbezügen wie "und dann dorthin navigieren" fragt
Noah nach der vollständigen Anfrage.

## Limits: what Noah cannot do

Noah cannot move money, make payments or transfers, change or close your
account, or reset your password. Noah is not a general-purpose assistant: it
does not answer questions about the weather, the news, translations, or any
topic outside shopping and your PayTo wallet, and it says so rather than
guessing. Noah does not see your name, email or payment details, and never
stores payment information.

Noah kann kein Geld überweisen, keine Zahlungen auslösen, dein Konto nicht
ändern oder löschen und kein Passwort zurücksetzen. Noah ist kein
Allzweck-Assistent: Fragen zu Wetter, Nachrichten oder Übersetzungen beantwortet
Noah nicht, sondern sagt das offen. Noah sieht weder deinen Namen noch deine
E-Mail oder Zahlungsdaten und speichert keine Zahlungsinformationen.
