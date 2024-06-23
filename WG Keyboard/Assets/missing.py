words = []
    

for line in open("./WGKeyboard-master/Assets/lexicon/lexicon.txt", "r", encoding="utf8"):    
    #print(f"<{line.strip("\n\t ")}>, {len(line.strip("\n\t "))}")
    words.append(line.strip("\n\t "))

words = list(set(words))
#print(words)

for line in open("MacKenziePhraseSet.txt", "r", encoding="utf8").readlines():    
    for word in line.lower().split(" "):
        word = word.strip("\n\t ")
        if word not in words:
            print(f"Phrase {line} has the word not present in the dictionary: <{word}>!")