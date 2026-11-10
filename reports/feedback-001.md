# Yaniv's listening feedback on the first listening page (verbatim), with decoding

Letter mapping used on the page: A = v2_default, B = v3_default, C = v3_loose, D = v3_cooler, E = v3_strict.
Set 1 = English: Version 1 en_basic (original reference), Version 2 en_basic_ref2 (new reference), Version 3 en_basic_round1.
Set 3 = Hebrew intro: Version 1 newer render (he_basic), Version 2 first round (he_basic_round1).
Set 4 = base vs trained (yaniv_v1 adapter): "second version" is the trained one for Hebrew; the English clip was trained only.

> "For set 1, all of the 3 versions says 99.5 MA and not FM. The tone is best on the 3rd option, but still sounding very computerized and not human.
>
> Set 2 sentence one d is the best one e and a are also pretty good.
>
> Sentence 2 it's d then a and b.
>
> On sentence number 3 only c sounds fair the others are far from correct pronouncing.
>
> On set number three the second version sound a bit better but both are far from sounding good with many mistakes in Hebrew and bad pronouncing it sounds like a machine because it's not emphasizing the correct words doesn't have good stops and some bad accent is coming in some words.
>
> On set for the second version sound more like me than the other recordings but still there is a tone problem intonation is not correct and some bad accent is coming in.
>
> The English for set for sound pretty similar to my voice much better than the current chatterbox model with my voice but it misses the energy and the drive of a radio DJ and we still have accent and intonation issues."

## Decoded
- Set 1: all three English versions say "99.5 MA" instead of "FM" (the checker saw "M8"/"mm"; the real failure is the spoken letters "FM"). Best tone: round-1 version (Version 3), still "computerized".
- Set 2 (Hebrew settings, blind): sentence 1: D (v3_cooler) best, E (v3_strict) and A (v2_default) also good. Sentence 2: D, then A, then B (v3_default). Sentence 3: only C (v3_loose) fair; A, B, D, E far from correct pronunciation. So D = v3_cooler is the most consistent, but no setting is reliable.
- Set 3: the first-round Hebrew intro (Version 2) sounds slightly better than the newer one. Both have many Hebrew mistakes, wrong emphasis, no good stops (pauses), and a foreign accent on some words.
- Set 4 (trained add-on yaniv_v1): the trained Hebrew sounds more like him than the other recordings, but intonation is wrong and some foreign accent remains. The trained English is clearly closer to his voice than the plain model, but lacks radio-DJ energy and drive; accent and intonation issues remain.
