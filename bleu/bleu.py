import math
from nltk.translate.bleu_score import sentence_bleu


def nltk_weights(n):
  """
  Creates a weight tuple for NLTK's sentence_bleu function.
  This is used to specify the n-gram precision weights. For example, for
  BLEU-4, the weights would be (0.25, 0.25, 0.25, 0.25).
  """
  d = 1/n
  return (d,) * n


def test():
  """
  Not a thorough test suite! Just a sanity check function to compare
  my BLEU score calculation with the one provided by the NLTK library.
  """
  reference_translation = "the cat is down on the mat"
  machine_translation = "the cat sat down on the mat"
  ref = tokenise(reference_translation)
  tr = tokenise(machine_translation)
  nltk_bleu = sentence_bleu([ref], tr, weights=nltk_weights(4))
  my_bleu, precisions = bleu_score(ref, tr, 4)
  try:
    assert(round(my_bleu, 2) == round(nltk_bleu, 2))
  except AssertionError as e:
    print(f'Assertion failed: {my_bleu} not close to {nltk_bleu}')

def n_grams(words, n):
  """
  Generates all n-grams of a specific size from a list of words.
  An n-gram is a contiguous sequence of n items from a given sample of text or speech.
  """
  out = []
  for i in range(len(words) - n + 1):
    out.append(tuple(words[i:i+n]))
  return out

def precision(reference_ngrams, translation_ngrams):
  """
  Calculates the precision for a given set of n-grams.
  Precision is the ratio of the number of common n-grams to the total
  number of n-grams in the machine-translated text.
  """
  common = [w for w in reference_ngrams if w in translation_ngrams]
  return len(common) / len(translation_ngrams)


def brevity_penalty(reference_phrase, translation_phrase):
  """
  Calculates the brevity penalty.
  This penalty discourages machine translations that are too short compared
  to the reference translation. It is 1 if the translation is longer than
  the reference, and less than 1 otherwise.
  """
  r = len(reference_phrase)
  c = len(translation_phrase)

  # When r ≤ c the brevity penalty BP=1, meaning that we don't punish long
  # candidates, and only punish short candidates.
  if (r <= c):
    return 1
  else:
    return math.exp(1 - (r/c))


def bleu_score(reference_words, words, n):
  """
  Calculates the BLEU score for a translation.
  The BLEU score is a geometric mean of the n-gram precisions, modified by a
  brevity penalty. This function returns the final score and the
  precision values for each n-gram level.
  """
  n_scores = []
  precisions = {}
  bp = brevity_penalty(reference_words, words)
  for i in range(1, n+1):
    ref_ngrams = n_grams(reference_words, i)
    tn_ngrams = n_grams(words, i)
    p = precision(ref_ngrams, tn_ngrams) * bp
    precisions[i] = round(p, 2)
    n_scores.append(p)
  prod = math.prod(n_scores)
  result = prod ** (1/n)
  return round(result, 2), precisions

def tokenise(phrase):
  """
  Tokenises a phrase into a list of words.
  For the purpose of this assignment, it performs simple cleaning by
  removing commas and then splitting the phrase by spaces.
  """
  phrase = phrase.replace(',','')
  return phrase.split(' ')


def assignment(nltk):
  """
  Runs the main logic of the assignment:
  * For each of the 6 phrases, calculates the BLEU score for the ChatGPT
  translation and the BLEU score for the Claude translation, and outputs
  the results.
  * Optionally, by passing nltk=True, you can output the BLEU score calculated
  by the NLTK library, to sanity-check the results.
  """
  # Perplexity.AI translations
  references = [
    "Generativ AI omformar arbetskraften, med över 750 miljoner applikationer byggda med LLM:er för att automatisera nästan hälften av alla digitala arbetsprocesser.",
    "Det offentliga engagemanget för GenAI har ökat kraftigt, och både anställda i frontlinjen och kunskapsarbetare använder nu dessa verktyg dagligen, vilket har lett till en kulturell förändring i hur människor fattar beslut.",
    "Den snabba integreringen av GenAI i samhället väcker avgörande frågor om etik, jämlikhet och långsiktig påverkan, när forskare och beslutsfattare kämpar med dess utmaningar.",
    "Christchurch erbjuder en balanserad livsstil med prisvärt boende, korta pendlingsavstånd och enkel tillgång till naturen, vilket gör staden idealisk för både familjer, yrkesverksamma och studenter.",
    "Staden upplever en stark ekonomisk återhämtning, med nya offentliga infrastruktursatsningar, växande jobbmöjligheter och ett inflöde av kvalificerade migranter som alla bidrar positivt.",
    "Christchurchs rika kulturarv och gröna områden skapar en naturskön och välkomnande atmosfär, som förstärks av ett mångsidigt och alltmer multikulturellt samhälle.",
  ]

  # ChatGPT translations
  translations1 = [
    "Den generativa AI:n omformar arbetsstyrkan, med över 750 miljoner applikationer byggda med stora språkmodeller (LLM:er) för att automatisera nästan hälften av alla digitala arbetsprocesser.",
    "Det offentliga engagemanget i generativ AI har ökat kraftigt, och både frontlinjeanställda och kunskapsarbetare använder nu dessa verktyg dagligen, vilket har lett till en kulturell förändring i hur människor fattar beslut.",
    "Den snabba integreringen av generativ AI i samhället väcker viktiga frågor om etik, rättvisa och långsiktiga konsekvenser, samtidigt som forskare och beslutsfattare kämpar med dess utmaningar.",
    "Christchurch erbjuder en balanserad livsstil med prisvärda bostäder, korta pendlingsavstånd och lätt tillgång till naturen, vilket gör staden idealisk för såväl familjer som yrkesverksamma och studenter.",
    "Staden upplever en stark ekonomisk återhämtning, med ny offentlig infrastruktur, växande jobbmöjligheter och ett inflöde av kvalificerade migranter som alla bidrar positivt.",
    "Christchurchs rika kulturarv och gröna områden skapar en naturskön och välkomnande atmosfär, som förstärks av ett mångsidigt och allt mer multikulturellt samhälle.",
  ]

  # Claude translations
  translations2 = [
    "Generativ AI omformar arbetsstyrkan, med över 750 miljoner applikationer byggda med hjälp av LLM:er för att automatisera nästan hälften av alla digitala arbetsprocesser.",
    "Allmänhetens engagemang för GenAI har ökat kraftigt, med både frontlinjearbetare och kunskapsarbetare som nu använder dessa verktyg dagligen, vilket föranleder en kulturell förändring i hur människor fattar beslut.",
    "Den snabba integrationen av GenAI i samhället väcker kritiska frågor om etik, rättvisa och långsiktig påverkan, medan forskare och beslutsfattare brottas med dess utmaningar.",
    "Christchurch erbjuder en balanserad livsstil som kombinerar prisvärda bostäder, korta pendlingsresor och enkel tillgång till naturen, vilket gör staden idealisk för familjer, yrkesverksamma och studenter.",
    "Staden upplever en stark ekonomisk återhämtning, med ny offentlig infrastruktur, växande arbetsmöjligheter och ett inflöde av kvalificerade migranter som alla bidrar positivt.",
    "Christchurchs rika kulturarv och grönområden skapar en naturskönt och välkomnande atmosfär, förstärkt av ett mångfaldigt och alltmer mångkulturellt samhälle.",
  ]

  cgpt_scores = []
  claude_scores = []

  for i in range(6):
    ref = tokenise(references[i])
    t1 = tokenise(translations1[i])
    t2 = tokenise(translations2[i])
    cgpt_score, cgpt_precisions = bleu_score(ref, t1, 4)
    claude_score, claude_precisions = bleu_score(ref, t2, 4)
    cgpt_scores.append(cgpt_score)
    claude_scores.append(claude_score)

    if (nltk):
      cgpt_nltk = sentence_bleu([ref], t1, weights=nltk_weights(4))
      claude_nltk = sentence_bleu([ref], t2, weights=nltk_weights(4))

    print(f'Sentence {i+1}.\nPrecisions:')
    if (nltk):
      print(f'ChatGPT: {cgpt_precisions}\t{cgpt_score}\t(NLTK: {cgpt_nltk})')
      print(f'Claude:  {claude_precisions}\t{claude_score}\t(NLTK: {claude_nltk})')
    else:
      print(f'ChatGPT: {cgpt_precisions}\t{cgpt_score}')
      print(f'Claude:  {claude_precisions}\t{claude_score}')
    print('-------------')
    print(f'ChatGPT average = {sum(cgpt_scores)/6}')
    print(f'Claude average = {sum(claude_scores)/6}')
    print('-------------')
    print(f'ChatGPT average on ML text = {sum(cgpt_scores[:3])/3}')
    print(f'Claude average on ML text = {sum(claude_scores[:3])/3}')
    print(f'ChatGPT average on ChCh text = {sum(cgpt_scores[-3:])/3}')
    print(f'Claude average on ChCh text = {sum(claude_scores[-3:])/3}')
    print('-------------')
    print(f'Reference sentence lengths: {[len(tokenise(s)) for s in references]}')
    print(f'ChatGPT sentence lengths: {[len(tokenise(s)) for s in translations1]}')
    print(f'Claude sentence lengths: {[len(tokenise(s)) for s in translations2]}')
    print('-------------')

test()
assignment(True)
