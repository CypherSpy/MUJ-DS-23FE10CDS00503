"""Script to generate and validate all project dataset files."""

import json
from pathlib import Path

import pandas as pd

from hinglishlens.tokenizer import tokenize

DATA_DIR = Path("data")
DATA_DIR.mkdir(parents=True, exist_ok=True)

# 1. sample_inputs.txt (15 sentences)
sample_sentences = [
    "yaar ye movie bakwaas thi but acting mast thi",
    "Aaj ka match toh ekdum zabardast tha, Kohli ne kamaal kar diya 🔥",
    "bhai delivery 3 din late aayi aur packing bhi kharab thi, worst service",
    "kyaa baat hai! mujhe toh bohot pasand aaya yeh phone",
    "wah bhai wah, 2 ghante wait karwaya aur bola table nahi hai 👏",
    "kal meeting subah 10 bje h, agenda mail kr diya",
    "Bohot hi accha product hai, battery backup is amazing!",
    "Khana bilkul thanda tha aur quantity bhi kam thi, total disappointment.",
    "Kya aap bata sakte hain ki flight status kya hai?",
    "Plz tmrw subah call karna, busy hu abhi.",
    "itna slow internet hai, 10 min se video load nahi ho raha 😡",
    "Awesome experience! Zomato order time par aa gaya.",
    "Delhi me monsoon kab tak aayega?",
    "Great job team, presentation ekdum badiya thi!",
    "Zabardast sarcasm, 5 star rating for ruining my evening!"
]

with open(DATA_DIR / "sample_inputs.txt", "w", encoding="utf-8") as f:
    f.write("\n".join(sample_sentences) + "\n")
print("Wrote sample_inputs.txt")

# 2. sentiment_train.csv (150 rows: 50 pos, 50 neg, 50 neu)
train_pos = [
    ("yeh phone bohot accha hai, battery backup is great", "positive"),
    ("khana ekdum swadisht tha aur service fast thi", "positive"),
    ("kohli ne aaj zabardast century mari 🏏", "positive"),
    ("movie ka climax mind-blowing tha, must watch!", "positive"),
    ("mumbai trip bohot badiya rahi, hotels were clean", "positive"),
    ("delivery time par aayi aur packaging perfect thi", "positive"),
    ("laptop performance ekdum smooth hai, love it", "positive"),
    ("teacher ne sab kuch ache se samjhaya, thank you", "positive"),
    ("iss app ka UI super smooth aur clean hai", "positive"),
    ("ye restaurant hamesha top notch food deta hai", "positive"),
    ("bohot hi pyara dress hai, fitting is perfect 😍", "positive"),
    ("flipkart discount offer bohot mast tha", "positive"),
    ("gameplay graphics bohot high quality hain", "positive"),
    ("gaana sunke maza aa gaya, super singer!", "positive"),
    ("flight journey smooth and comfortable thi", "positive"),
    ("ye book padhke maza aaya, highly recommended", "positive"),
    ("hotel staff bohot polite aur helpful tha", "positive"),
    ("camera quality in low light is awesome 🔥", "positive"),
    ("subah ki chai aur ye mausam, dil khush ho gaya", "positive"),
    ("support team ne issue 5 min me resolve kar diya", "positive"),
    ("sharma ji ka ladka topper ban gaya, so proud", "positive"),
    ("laptop cooling fan is very quiet and effective", "positive"),
    ("ye shoes bohot comfortable hain for running", "positive"),
    ("festival offers me acchi deal mil gayi", "positive"),
    ("sound quality of these earphones is crystal clear", "positive"),
    ("fast charging 30 min me full charge kar deta hai", "positive"),
    ("bhai kya mast biryani bani hai aaj 🍲", "positive"),
    ("cab driver bohot friendly aur safe driving kar raha tha", "positive"),
    ("exam me saare questions aate the, awesome result", "positive"),
    ("bengaluru me weather aaj bohot pleasant hai", "positive"),
    ("cleanliness and hygiene standard bohot badiya hai", "positive"),
    ("new update me saare bugs fix ho gaye hain", "positive"),
    ("customer support was super quick and responsive", "positive"),
    ("film ka background music ekdum top level hai", "positive"),
    ("aaj mera birthday bohot special raha, thanks everyone", "positive"),
    ("gym instructor ne bohot acchi training di", "positive"),
    ("ye dessert khake maza aa gaya, so delicious", "positive"),
    ("air conditioner cooling bohot fast karta hai", "positive"),
    ("project presentation me client bohot happy hua", "positive"),
    ("online shopping experience was fast and hassle-free", "positive"),
    ("aaj ki morning walk se bohot energetic feel ho raha hai", "positive"),
    ("ye mechanical keyboard typing ke liye best hai", "positive"),
    ("train time par destination pahunch gayi", "positive"),
    ("bhai ne surprise gift diya, bohot khush hu", "positive"),
    ("iss cafe ki cold coffee is absolute heaven ❤️", "positive"),
    ("doctor ne bohot ache se consult kiya, feel much better", "positive"),
    ("wifi speed is super fast today ⚡", "positive"),
    ("concert performance energetic aur thrilling thi", "positive"),
    ("garden me phool dekhke mood accha ho gaya", "positive"),
    ("everything was handled smoothly, full satisfaction", "positive")
]

train_neg = [
    ("itna bekaar phone aaj tak nahi dekha, waste of money", "negative"),
    ("khana ekdum bakwaas aur thanda tha 🤮", "negative"),
    ("delivery 4 din late aayi, horrible service", "negative"),
    ("app baar baar crash ho raha hai, fix karo", "negative"),
    ("customer care koi response nahi deta, worst support", "negative"),
    ("movie bilkul boring aur timepass thi, total waste", "negative"),
    ("product broken condition me mila, refund chahiye", "negative"),
    ("hotel room me bohot smell thi aur washroom dirty tha", "negative"),
    ("network coverage bilkul zero hai iss sim card me", "negative"),
    ("laptop 10 min me overheat ho jata hai", "negative"),
    ("price bohot high hai lekin quality ekdum cheap", "negative"),
    ("cab cancel kar di driver ne after making me wait 20 min", "negative"),
    ("flight delay ho gayi 3 ghante, no updates given", "negative"),
    ("earphones ka left side 2 din me kaam karna band kar gaya", "negative"),
    ("khana bilkul oily aur salty tha, ruined my dinner", "negative"),
    ("ye update ke baad phone lag ho raha hai", "negative"),
    ("seller fraud hai, fake item bhej diya", "negative"),
    ("train station par hygiene bilkul zero hai", "negative"),
    ("battery 2 ghante me khatam ho jaati hai, pathetic", "negative"),
    ("service staff bohot rude aur arrogant tha", "negative"),
    ("game me bohot saare bugs aur glitches hain", "negative"),
    ("pizza box damaged tha aur pizza thanda tha 🍕👎", "negative"),
    ("payment deduct ho gaya lekin order confirm nahi hua", "negative"),
    ("shirt size wrong aaya aur return option nahi dikh raha", "negative"),
    ("sound volume is too low even at max level", "negative"),
    ("itna ganda experience pehle kabhi nahi hua", "negative"),
    ("ac se paani leak ho raha hai, useless repairman", "negative"),
    ("movie theatre me AC kaam nahi kar raha tha", "negative"),
    ("traffic jam me 2 ghante waste ho gaye", "negative"),
    ("bill me extra charges lagaye bina bataye", "negative"),
    ("pant material is very thin and cheap quality", "negative"),
    ("website load hi nahi ho rahi, server error 500", "negative"),
    ("wah kya baat hai, 1 hr wait ke baad cold food diya 👏", "negative"),
    ("great service, my order was lost by delivery boy 😒", "negative"),
    ("keyboard keys stuck ho rahi hain continuously", "negative"),
    ("doctor was extremely late and in a hurry", "negative"),
    ("electricity cut for 5 hours in hot summer", "negative"),
    ("mechanic took money but problem not fixed", "negative"),
    ("hotel breakfast spread was stale and limited", "negative"),
    ("shoes torn within 1 week of normal use", "negative"),
    ("bank app transactions failed multiple times", "negative"),
    ("movie script completely senseless and stupid", "negative"),
    ("car mileage is very poor compared to claim", "negative"),
    ("water purifier filter expired in 1 month", "negative"),
    ("bus driver driving very rashly, scary journey", "negative"),
    ("printer keeps jamming every 2 pages", "negative"),
    ("superb management, 3 hours queue for 1 entry pass 👏", "negative"),
    ("package packaging was completely crushed", "negative"),
    ("noise cancellation in headphones is useless", "negative"),
    ("terrible quality product, don't buy this trash", "negative")
]

train_neu = [
    ("kal subah 9 bje meeting scheduled hai", "neutral"),
    ("is product ki warranty kitne saal ki hai?", "neutral"),
    ("train schedule check kar lo IRCTC app par", "neutral"),
    ("delhi me aaj temperature 35 degree celsius hai", "neutral"),
    ("meeting room number 3 me conference chal rahi hai", "neutral"),
    ("kya aap ye file email par send kar sakte hain?", "neutral"),
    ("dukaan kal subah 10 bje khulegi", "neutral"),
    ("parcel dispatch ho gaya hai, tracking number 12345", "neutral"),
    ("aaj shaam 7 bje cricket match start hoga", "neutral"),
    ("kya iss phone me 5G support available hai?", "neutral"),
    ("flight schedule is updated on the website", "neutral"),
    ("office shift timing 9 AM to 6 PM hai", "neutral"),
    ("kya mumbai me aaj rain alert hai?", "neutral"),
    ("exam schedule official portal par release ho gaya hai", "neutral"),
    ("payment successful, invoice attached", "neutral"),
    ("kya aapka office sunday ko open hota hai?", "neutral"),
    ("battery capacity is 5000 mAh with 33W charger", "neutral"),
    ("hiring notice shared on linkedin page", "neutral"),
    ("system update size is 1.2 GB", "neutral"),
    ("restaurant timing is 11 am to 11 pm daily", "neutral"),
    ("kya metro station yahan se door hai?", "neutral"),
    ("bus stand next crossroad par hai", "neutral"),
    ("class timing change hoke 2 PM ho gayi hai", "neutral"),
    ("check-in time is 12 PM and check-out is 10 AM", "neutral"),
    ("bank holiday on upcoming monday due to festival", "neutral"),
    ("ticket booking opens tomorrow at 10 AM", "neutral"),
    ("kya table reservation required hai for tonight?", "neutral"),
    ("documents verify karke submit kar diye hain", "neutral"),
    ("next station is rajiv chowk", "neutral"),
    ("weather forecast says cloudy skies for next 2 days", "neutral"),
    ("item dispatched from gurgaon warehouse", "neutral"),
    ("kya ye laptop Windows 11 pre-installed aata hai?", "neutral"),
    ("wifi password is written on the router", "neutral"),
    ("store location is near main market circle", "neutral"),
    ("meeting link share kar diya hai chat room me", "neutral"),
    ("admission process details website par available hain", "neutral"),
    ("kya aap mujhe address text kar sakte ho?", "neutral"),
    ("lunch break starts at 1:30 PM", "neutral"),
    ("speed limit on expressway is 100 km/h", "neutral"),
    ("parking facility is available in basement 2", "neutral"),
    ("kya isme dual sim slot milta hai?", "neutral"),
    ("subscription renewal date is next month", "neutral"),
    ("passport verification completed at police station", "neutral"),
    ("kya train me pantry car service hai?", "neutral"),
    ("library is open on weekdays till 8 PM", "neutral"),
    ("otp generated for transaction reference 9876", "neutral"),
    ("lecture hall 4 me class start hone wali hai", "neutral"),
    ("kya courier service sunday ko delivery karti hai?", "neutral"),
    ("gas cylinder booking successful, reference id 4567", "neutral"),
    ("flight duration from delhi to mumbai is 2 hours", "neutral")
]

train_df = pd.DataFrame(train_pos + train_neg + train_neu, columns=["text", "label"])
train_df.to_csv(DATA_DIR / "sentiment_train.csv", index=False)
print(f"Wrote sentiment_train.csv ({len(train_df)} rows)")

# 3. sentiment_test.csv (60 rows: 20 pos, 20 neg, 20 neu)
test_pos = [
    ("aaj ki morning coffee ke sath weather ekdum perfect hai ☕", "positive"),
    ("bhai flipkart sale me mast discount mila", "positive"),
    ("movie directional work and screenplay super badiya tha", "positive"),
    ("khana bohot yummy tha, full family enjoyed", "positive"),
    ("phone camera daylight photos me clarity unbelievable hai", "positive"),
    ("delivery boy reached in 15 mins, super quick!", "positive"),
    ("ye concert lifetime memory ban gaya, epic energy 🎉", "positive"),
    ("hotel room setup and view was amazing", "positive"),
    ("bhai ne exam clear kar liya, super happy news", "positive"),
    ("customer care team resolved query instantly, top rating", "positive"),
    ("laptop boots in 5 seconds, extremely fast SSD", "positive"),
    ("iss cafe ka ambiance and playlist bohot chill hai", "positive"),
    ("shirt fitting and fabric quality is outstanding", "positive"),
    ("cricket match me india ki victory super exciting thi", "positive"),
    ("driver drove smoothly and reached before time", "positive"),
    ("software update made interface much faster", "positive"),
    ("ye dessert dish ekdum lajawab thi", "positive"),
    ("doctor was very polite and explained treatment clearly", "positive"),
    ("earphones bass and treble balance is perfect", "positive"),
    ("fast delivery and genuine product, highly satisfied", "positive")
]

test_neg = [
    ("3 ghante se internet band hai, useless network", "negative"),
    ("khana ekdum tasteless aur stale bhej diya", "negative"),
    ("worst customer support experience, no resolution provided", "negative"),
    ("screen 2 din me automatic crack ho gayi, pathetic build", "negative"),
    ("delivery delay of 5 days, seller not responding", "negative"),
    ("movie total headache thi, money wasted", "negative"),
    ("cab booked thi lekin driver ne call cut karke cancel kar diya", "negative"),
    ("room me AC loud noise kar raha tha aur cooling zero", "negative"),
    ("app keeps crashing on checkout page, worst bug", "negative"),
    ("food item missing from delivery parcel 😡", "negative"),
    ("battery draining 20 percent per hour without use", "negative"),
    ("shoes quality is very cheap, sole detached in 3 days", "negative"),
    ("flight delayed by 4 hours without any notification", "negative"),
    ("wah kya packaging hai, pure glass item ko break kar diya 👏", "negative"),
    ("sound distorted at high volume, useless speaker", "negative"),
    ("refund request rejected without any valid explanation", "negative"),
    ("hotel washroom water supply was not working", "negative"),
    ("terrible experience at store, staff behaviour very arrogant", "negative"),
    ("train compartment was dirty and overcrowded", "negative"),
    ("keyboard keys malfunctioning right out of box", "negative")
]

test_neu = [
    ("kya kal office me working day hai ya holiday?", "neutral"),
    ("train arrival timing is 8:30 AM at platform 2", "neutral"),
    ("parcel location showing in transit near delhi hub", "neutral"),
    ("kya is smart watch me heart rate sensor hai?", "neutral"),
    ("meeting call scheduled for 4 PM today", "neutral"),
    ("exam center address is updated on admit card", "neutral"),
    ("kya restaurant me outdoor seating available hai?", "neutral"),
    ("bank branch working hours are 10 AM to 4 PM", "neutral"),
    ("kya laptop box me HDMI cableIncluded hai?", "neutral"),
    ("flight status is currently on time", "neutral"),
    ("store opens at 10 AM every morning", "neutral"),
    ("kya ye bus direct jaipur jayegi?", "neutral"),
    ("order reference number is 987654321", "neutral"),
    ("weather update predicts light rain in evening", "neutral"),
    ("kya registration process me aadhaar card required hai?", "neutral"),
    ("class notes shared on google drive folder", "neutral"),
    ("next metro halt is Huda City Centre", "neutral"),
    ("kya sunday ko clinic open hota hai?", "neutral"),
    ("package dimensions are 30cm by 20cm", "neutral"),
    ("payment transaction id sent via SMS", "neutral")
]

test_df = pd.DataFrame(test_pos + test_neg + test_neu, columns=["text", "label"])
test_df.to_csv(DATA_DIR / "sentiment_test.csv", index=False)
print(f"Wrote sentiment_test.csv ({len(test_df)} rows)")

# 4. lid_gold.jsonl (25 sentences with exact token-level language tags)
gold_data = [
    ("yaar ye movie bakwaas thi", ["HI", "HI", "EN", "HI", "HI"]),
    ("Kohli scored 100 runs today", ["NE", "EN", "UNIV", "EN", "EN"]),
    ("mujhe to pasand hai", ["HI", "HI", "HI", "HI"]),
    ("bhai delivery late aayi", ["HI", "EN", "EN", "HI"]),
    ("Zomato order fast tha", ["NE", "EN", "EN", "HI"]),
    ("Aaj ka match toh ekdum zabardast tha", ["HI", "HI", "EN", "HI", "HI", "HI", "HI"]),
    ("plz tmrw subah call karna", ["EN", "EN", "HI", "EN", "HI"]),
    ("worst service ever , total waste", ["EN", "EN", "EN", "UNIV", "EN", "EN"]),
    ("kal meeting 10 bje h", ["HI", "EN", "UNIV", "HI", "HI"]),
    ("kyun bhai kya hua ?", ["HI", "HI", "HI", "HI", "UNIV"]),
    ("Jaipur trip was awesome", ["NE", "EN", "EN", "EN"]),
    ("achha ji aisa hai kya", ["HI", "HI", "HI", "HI", "HI"]),
    ("very good product , love it !", ["EN", "EN", "EN", "UNIV", "EN", "EN", "UNIV"]),
    ("kya baat hai !", ["HI", "HI", "HI", "UNIV"]),
    ("network issue hai yahan", ["EN", "EN", "HI", "HI"]),
    ("Delhi me weather badiya hai", ["NE", "HI", "EN", "HI", "HI"]),
    ("bohot hi kharab experience", ["HI", "HI", "HI", "EN"]),
    ("thanks bro , mail kr diya", ["EN", "EN", "UNIV", "EN", "HI", "HI"]),
    ("kya kal match hoga ?", ["HI", "HI", "EN", "HI", "UNIV"]),
    ("superb acting by hero 👏", ["EN", "EN", "EN", "EN", "UNIV"]),
    ("battery backup is 100%", ["EN", "EN", "EN", "UNIV", "UNIV"]),
    ("mujhe lagta hai ye true hai", ["HI", "HI", "HI", "HI", "EN", "HI"]),
    ("great work done by team", ["EN", "EN", "EN", "EN", "EN"]),
    ("wah bhai wah 👏", ["HI", "HI", "HI", "UNIV"]),
    ("speed slow h , fix kr do", ["EN", "EN", "HI", "UNIV", "EN", "HI", "HI"])
]

# Verify token alignment for all gold sentences
validated_gold = []
for idx, (sentence, tags) in enumerate(gold_data):
    spans = tokenize(sentence)
    token_texts = [s.text for s in spans]
    if len(token_texts) != len(tags):
        print(f"ERROR on item {idx}: sentence '{sentence}'")
        print(f"Tokenized ({len(token_texts)}): {token_texts}")
        print(f"Tags ({len(tags)}): {tags}")
        raise ValueError("Token alignment mismatch in lid_gold dataset!")
    validated_gold.append({"text": sentence, "tags": tags})

with open(DATA_DIR / "lid_gold.jsonl", "w", encoding="utf-8") as f:
    f.writelines(json.dumps(item, ensure_ascii=False) + "\n" for item in validated_gold)
print(f"Wrote lid_gold.jsonl ({len(validated_gold)} rows - 100% token aligned)")
