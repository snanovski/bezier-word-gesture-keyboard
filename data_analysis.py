import numpy as np
import matplotlib.pyplot as plt

files_part_1 = ["user_01.txt", "user_02.txt", "user_03.txt", "user_04.txt", "user_05.txt", "user_06.txt"]
files_part_1.extend(["user_07.txt", "user_08.txt", "user_10.txt", "user_11.txt", "user_12.txt", "user_13.txt"])

files_part_2 = ["user_14.txt", "user_15.txt", "user_16.txt", "user_17.txt", "user_18.txt", "user_19.txt", "user_20.txt", "user_21.txt", "user_22.txt"]

files = files_part_1
#files = files_part_2
#files.extend(files_part_1)

bezier_data = []
bezier_wpm = 0.0
bezier_data_user = []
bezier_count = 0.0
bezier_ter_data = []
bezier_ter_data_user = []
line_data = []
line_wpm = 0.0
line_data_user = []
line_count = 0.0
line_ter_data = []
line_ter_data_user = []


all_data = []
all_ter_data = []
all_data_user = []
all_ter_data_user = []

full_data = []

bezier = True

removed_samples = 0

wpm_per_phrase_index = []
for _ in range(8):
    wpm_per_phrase_index.append([])

user_data_for_training = []

for file in files:
    cur_wpm = 0.0
    cur_count = 0.0    
    delete_streak = 0
    max_delete_streak = 0

    cur_phrase = ""
    graphs = []
    word_graph_idx = 0

    cur_sample_data = {}

    cur_user_bezier = []
    cur_user_line = []
    cur_user_both = []
    cur_user_ter_bezier = []
    cur_user_ter_line = []
    cur_user_ter_both = []

    cur_phrase_idx = -1

    mistakes = 0
    last_deleted = False

    previous_command = ""

    mistake_word_indexes = {}

    for line in open(f"./Data/{file}", "r").readlines():        
        parts = line.strip().split(":")                

        if previous_command == "POINTS" and parts[0] != "WORD":
            mistakes += 1

        if parts[0] != "WORD" and previous_command == "DELETED":
            mistakes += 1

        if parts[0] == 'WORD' and parts[2].isdigit():            
            mistakes -= 1

        last_deleted = False
        if parts[0] == "DELETED":
            delete_streak += 1
            #mistakes += 1
            #last_deleted = True
            max_delete_streak = max(max_delete_streak, delete_streak)
        else:
            delete_streak = 0

        if (parts[0] == "===START==="):
            mistakes = 0
            delete_streak = 0
            max_delete_streak = 0        
            cur_sample_data = {}
        elif (parts[0] == "PHRASE"):            
            cur_phrase = parts[2]      
            graphs = ["_" for _ in range(len(cur_phrase.split(" ")))]
            word_graph_idx = 0
            cur_phrase_idx += 1
        elif (parts[0] == "WORD"):
            word_graph_idx += 1
        elif (parts[0] == "DELETED"):
            word_graph_idx -= 1
        elif (parts[0] == "POINTS"):
            if (word_graph_idx < len(graphs) and 0 <= word_graph_idx):
                graphs[word_graph_idx] = parts[2]
        elif parts[0] == "ALGORITHM":            
            bezier = (parts[2] == "Cubic Bezier")
        elif parts[0] == "WPM":
            if (max_delete_streak >= 2):
                removed_samples += 1
                previous_command = parts[0]
                continue
            wpm = float(parts[2])        
            cur_sample_data["WPM"] = wpm                
            cur_wpm += wpm
            cur_count += 1.0
            #if (cur_count <= 8):
            #    continue             
        
            cur_sample_data["WORDS"] = []
            for i, word in enumerate(cur_phrase.split(" ")):            
                user_data_for_training.append(f"{word}:{graphs[i]}")
                cur_sample_data["WORDS"].append((word, graphs[i]))

            words_in_phrase = len(cur_sample_data["WORDS"])
            cur_sample_data["TER"] = (mistakes)/(words_in_phrase + mistakes)            
            

            all_data.append(wpm)
            cur_user_both.append(wpm)
            all_ter_data.append(cur_sample_data["TER"])
            wpm_per_phrase_index[cur_phrase_idx // 2].append(wpm)
            cur_user_ter_both.append(cur_sample_data["TER"])
            if (bezier):
                bezier_data.append(wpm)                
                bezier_ter_data.append(cur_sample_data["TER"])
                cur_user_bezier.append(wpm)
                cur_user_ter_bezier.append(cur_sample_data["TER"])
            else:
                line_data.append(wpm)
                line_ter_data.append(cur_sample_data["TER"])
                cur_user_line.append(wpm)
                cur_user_ter_line.append(cur_sample_data["TER"])
        
        previous_command = parts[0]
                           
    bezier_data_user.append(np.mean(cur_user_bezier))
    line_data_user.append(np.mean(cur_user_line))
    all_data_user.append(np.mean(cur_user_both))

    bezier_ter_data_user.append(np.mean(cur_user_ter_bezier))
    line_ter_data_user.append(np.mean(cur_user_ter_line))
    all_ter_data_user.append(np.mean(cur_user_ter_both))

    print(f"For file = {file}: WPM = {cur_wpm / cur_count}")

# print(user_data_for_training[0])
f = open("./Data/user_training_data_first_half.txt", "a")
for line in user_data_for_training:
    f.write(f"{line}\n")
f.close()

print(f"Removed samples = {removed_samples}")

print(f"Bezier WPM: Mean = {sum(bezier_data) / float(len(bezier_data))}, Std = {np.std(bezier_data)}, Count = {len(bezier_data)}")
print(f"Bezier WPM Per User: Mean = {sum(bezier_data_user) / float(len(bezier_data_user))}, Std = {np.std(bezier_data_user)}, Count = {len(bezier_data_user)}")
print(f"Bezier TER: Mean = {sum(bezier_ter_data) / float(len(bezier_ter_data))}, Std = {np.std(bezier_ter_data)}, Count = {len(bezier_ter_data)}")
print(f"Bezier TER Per User: Mean = {sum(bezier_ter_data_user) / float(len(bezier_ter_data_user))}, Std = {np.std(bezier_ter_data_user)}, Count = {len(bezier_ter_data_user)}")

#print(bezier_data)
print(f"Line WPM: Mean = {sum(line_data) / float(len(line_data))}, Std = {np.std(line_data)}, Count = {len(line_data)}")
print(f"Line WPM Per User: Mean = {sum(line_data_user) / float(len(line_data_user))}, Std = {np.std(line_data_user)}, Count = {len(line_data_user)}")
print(f"Line TER: Mean = {sum(line_ter_data) / float(len(line_ter_data))}, Std = {np.std(line_ter_data)}, Count = {len(line_ter_data)}")
print(f"Line TER Per User: Mean = {sum(line_ter_data_user) / float(len(line_ter_data_user))}, Std = {np.std(line_ter_data_user)}, Count = {len(line_ter_data_user)}")
#print(line_data)
print(f"All WPM: Mean = {sum(all_data) / float(len(all_data))}, Std = {np.std(all_data)}, Count = {len(all_data)}")
print(f"All WPM Per User: Mean = {sum(all_data_user) / float(len(all_data_user))}, Std = {np.std(all_data_user)}, Count = {len(all_data_user)}")
print(f"All TER: Mean = {sum(all_ter_data) / float(len(all_ter_data))}, Std = {np.std(all_ter_data)}, Count = {len(all_ter_data)}")
print(f"All TER Per User: Mean = {sum(all_ter_data_user) / float(len(all_ter_data_user))}, Std = {np.std(all_ter_data_user)}, Count = {len(all_ter_data_user)}")

mean_wpm_per_phrase_index = [np.mean(x) for x in wpm_per_phrase_index]
print(mean_wpm_per_phrase_index)

plt.plot(range(2, 17, 2), mean_wpm_per_phrase_index)
plt.show()

# plt.boxplot([bezier_data, line_data, all_data], patch_artist=True, labels=["Bezier", "Line", "Both"])
# plt.show()
