import pyterrier as pt
import random
import pandas as pd
import os
import sys

# Dictionary of commonly misspelled words
misspellings = {
    "you're": "your",
    "there": "their",
    "they're": "their",
    "you": "u",
    "their": "thier",
    "receive": "recieve",
    "separate": "seperate",
    "definitely": "definately",
    "occurrence": "occurence",
    "accommodate": "acommodate",
    "privilege": "privelege",
    "government": "goverment",
    "existence": "existance",
    "embarrass": "embarass",
    "abscess": "abscsess",
    "abseil": "abseil",
    "accommodate": "acommodate",
    "accommodation": "acommodation",
    "accumulate": "acumulate",
    "accumulation": "acumulation",
    "achieve": "acheive",
    "acquaint": "aquaint",
    "acquire": "aquire",
    "acquit": "aquit",
    "address": "adress",
    "aggressive": "agresive",
    "aggression": "agresion",
    "all right": "alright",
    "a lot": "alot",
    "amateur": "amatuer",
    "anaesthetic": "anesthetic",
    "Antarctic": "Antartic",
    "apartment": "appartment",
    "apparent": "aparent",
    "aqueduct": "aquaduct",
    "archaeology": "archeology",
    "Arctic": "Artic",
    "argument": "arguement",
    "artefact": "artifact",
    "asterisk": "asterix",
    "attach": "atach",
    "beautiful": "beatiful",
    "belief": "beleif",
    "believe": "beleive",
    "benighted": "beniteed",
    "besiege": "beseige",
    "biased": "biassed",
    "bigoted": "biggotted",
    "blatant": "blatent",
    "brief": "breif",
    "broccoli": "brocoli",
    "buoy": "boyant",
    "buoyant": "boyent",
    "cappuccino": "capuccino",
    "Caribbean": "Carribean",
    "ceiling": "cieling",
    "cemetery": "cemetary",
    "civilian": "civillian",
    "coconut": "cocanut",
    "commemorate": "comemmorate",
    "commitment": "committment",
    "committee": "commitee",
    "comparative": "compairative",
    "compatible": "compatable",
    "competent": "compitant",
    "conceive": "concieve",
    "consensus": "concensus",
    "contemporary": "contempory",
    "correspondence": "correspondance",
    "cursor": "curser",
    "deceive": "decieve",
    "definite": "definate",
    "descendant": "descendent",
    "despair": "dispair",
    "desperate": "desparate",
    "detach": "detatch",
    "diarrhoea": "diarhea",
    "disappear": "dissapear",
    "disappoint": "dissapoint",
    "disastrous": "disasterous",
    "discipline": "disipline",
    "dissect": "disect",
    "ecstasy": "ecstacy",
    "eighth": "eigth",
    "embarrass": "embarass",
    "environment": "enviroment",
    "espresso": "expresso",
    "estuary": "estuery",
    "exaggerate": "exagerate",
    "except": "excpet",
    "exhilarate": "exhilirate",
    "existence": "existance",
    "extraordinary": "extrordinary",
    "extrovert": "extravert",
    "familiar": "familliar",
    "fascinate": "facinante",
    "February": "Febuary",
    "fierce": "feirce",
    "fluorescent": "florescent",
    "foreign": "foriegn",
    "forty": "fourty",
    "friend": "freind",
    "fulfil": "fulfill",
    "gauge": "guage",
    "glamorous": "glamerous",
    "government": "goverment",
    "graffiti": "grafitti",
    "grammar": "grammer",
    "grateful": "greatful",
    "grief": "greif",
    "guarantee": "garantee",
    "guard": "gaurd",
    "hamster": "hampster",
    "handkerchief": "hankerchief",
    "harass": "harrass",
    "hers": "her's",
    "hierarchy": "heirarchy",
    "hindrance": "hindrence",
    "homogeneous": "homogenous",
    "honorary": "honourary",
    "humorous": "humourous",
    "hygiene": "hygeine",
    "idiosyncrasy": "idiosyncracy",
    "imaginary": "imaginery",
    "immediately": "imediately",
    "inadvertent": "inadvertant",
    "independent": "independant",
    "inoculate": "innoculate",
    "insistent": "insistant",
    "instalment": "installment",
    "interrupt": "interupt",
    "irrelevant": "irrevelant",
    "itinerary": "itinery",
    "jocular": "joculer",
    "judgement": "judgment",
    "kernel": "kernal",
    "knowledge": "knowlege",
    "language": "languge",
    "liaise": "liase",
    "library": "libary",
    "lightning": "lightening",
    "liquefy": "liquify",
    "maintenance": "maintainance",
    "manoeuvre": "manuver",
    "medicine": "medecine",
    "Mediterranean": "Mediteranean",
    "millennium": "milennium",
    "millionaire": "millionare",
    "miniature": "minature",
    "minuscule": "miniscule",
    "mischievous": "mischevious",
    "misspell": "mispell",
    "moreover": "more over",
    "necessary": "neccessary",
    "niece": "neice",
    "noticeable": "noticable",
    "occasion": "ocassion",
    "occur": "ocur",
    "occurrence": "ocurrance",
    "omission": "ommision",
    "opportunity": "oppertunity",
    "parallel": "paralel",
    "parliament": "parliment",
    "perceive": "percieve",
    "permanent": "perminent",
    "persistent": "persistant",
    "pharaoh": "pharoah",
    "pigeon": "pidgeon",
    "Portuguese": "Portugese",
    "possess": "posess",
    "potato": "potatoe",
    "privilege": "priviledge",
    "pronunciation": "pronounciation",
    "questionnaire": "questionaire",
    "recommend": "reccomend",
    "refrigerator": "refridgerator",
    "relevant": "revelant",
    "rhythm": "rythm",
    "separate": "seperate",
    "siege": "seige",
    "sieve": "seive",
    "successful": "sucessful",
    "supersede": "supercede",
    "suppress": "supress",
    "surprise": "suprise",
    "tariff": "tarif",
    "temperature": "temprature",
    "theirs": "their's",
    "threshold": "threshhold",
    "tomorrow": "tommorow",
    "truly": "truely",
    "until": "untill",
    "weird": "wierd",
    "yield": "yeild",
    "yours": "your's"
}

def introduce_spelling_mistakes(text, drop_prob=0.1, swap_prob=0.2):
    """
    Randomly dropping letters from words and swapping commonly misspelled words with incorrect variants
    according to: Speech and Language Processing. Daniel Jurafsky & James H. Martin.
    "estimates for the frequency of spelling errors in human-typed text vary from 1-2% for carefully 
    retyping already printed text to 10-15% for web queries"
    """
    words = text.split()
    new_words = []

    for word in words:
        # Swap commonly misspelled words with their incorrect versions with probability swap_prob
        if word.lower() in misspellings and random.random() < swap_prob:
            new_words.append(misspellings[word.lower()])
        else:
            # Randomly drop letters with probability drop_prob
            new_word = "".join(
                [char for char in word if random.random() > drop_prob]
            )
            new_words.append(new_word if new_word else word)  # Ensure we don't delete whole words

    return " ".join(new_words)

def process_dataset(name):
    if not pt.java.started():
        pt.init()

    dataset = pt.get_dataset(name)
    queries = dataset.get_topics()
    queries["modified_query"] = queries["query"].apply(
        lambda q: introduce_spelling_mistakes(q, drop_prob=0.1, swap_prob=0.2)
    )
    queries[["qid", "query", "modified_query"]].to_csv(f"./{name.replace("/", "")}.csv", index=False)





if __name__ == "__main__":
    # Process datasets and introduce spelling mistakes
    process_dataset(str(sys.argv[1]))

 