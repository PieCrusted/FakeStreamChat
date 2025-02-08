import random
import sys

def load_emotes(file_path):
    with open(file_path, 'r') as file:
        emotes = file.read().splitlines()
    return emotes

def generate_random_messages(emotes, slangs, dist_emotes, dist_slang, dist_slang_emote, message_count):
    total_dist = dist_emotes + dist_slang + dist_slang_emote
    emotes_prob = dist_emotes / total_dist
    slang_prob = dist_slang / total_dist
    slang_emote_prob = dist_slang_emote / total_dist

    messages = []
    for _ in range(message_count):
        choice = random.choices(['emotes', 'slang', 'slang_emote'], [emotes_prob, slang_prob, slang_emote_prob])[0]
        if choice == 'emotes':
            emote = random.choice(emotes)
            messages.append(f"{emote} {emote} {emote}")
        elif choice == 'slang':
            slang = random.choice(slangs)
            messages.append(slang)
        elif choice == 'slang_emote':
            slang = random.choice(slangs)
            emote = random.choice(emotes)
            messages.append(f"{slang} {emote} {emote} {emote}")
    
    return messages

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python systematic_emote_spam.py <emote_file> [<dist_emotes> <dist_slang> <dist_slang_emote> <message_count>]")
        sys.exit(1)

    emote_file = sys.argv[1]
    emotes = load_emotes(emote_file)

    if len(sys.argv) == 3:
        dist_emotes, dist_slang, dist_slang_emote, message_count = 3, 3, 3, int(sys.argv[2])
    elif len(sys.argv) == 6:
        dist_emotes = int(sys.argv[2])
        dist_slang = int(sys.argv[3])
        dist_slang_emote = int(sys.argv[4])
        message_count = int(sys.argv[5])
    else:
        dist_emotes, dist_slang, dist_slang_emote, message_count = 3, 3, 3, 1

    # Common internet slang TODO: Add to the list
    # Also add pseudo priority by repeating slangs in the list for more common sighting
    slangs = [
        'LOL', 'lol', 'Lol', 'LMAO', 'BRUH', 'OMG', 'ROFL', 'SMH', 'YOLO', 'AFK', 'FR', 'GG', 'GLHF', 'JK', 'WTF', 'WTH',
        'LOL', 'lol', 'Lol', 'lmao', 'bruh', 'omg', 'rofl', 'smh', 'yolo', 'afk', 'fr', 'gg', 'glhf', 'js', 'wtf', 'wth',
        'LOL', 'lol', 'Lol', 'lmao', 'bruh', 'omg', 'rofl', 'smh', 'yolo', 'afk', 'fr', 'gg', 'glhf', 'js', 'wtf', 'wth',
        'LOL', 'lol', 'Lol', 'lmao', 'bruh', 'omg', 'rofl', 'smh', 'yolo', 'afk', 'fr', 'gg', 'glhf', 'js', 'wtf', 'wth'
    ]

    if emotes:
        random_messages = generate_random_messages(emotes, slangs, dist_emotes, dist_slang, dist_slang_emote, message_count)
        for message in random_messages:
            print(message)
    else:
        print("No emotes found in the file.")