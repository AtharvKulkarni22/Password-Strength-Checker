import re
import string
import math
from collections import Counter
from Levenshtein import distance as levenshtein_distance
import nltk
from nltk.corpus import words

nltk.download("words", quiet=True)
english_words = set(words.words())

# Top 100 Common passwords set. Source: https://en.wikipedia.org/wiki/Wikipedia:10,000_most_common_passwords
common_passwords = set([
    "123456", "password", "12345678", "qwerty", "123456789", "12345", "1234", "111111",
    "1234567", "dragon", "123123", "baseball", "abc123", "football", "monkey", "letmein",
    "696969", "shadow", "master", "666666", "qwertyuiop", "123321", "mustang", "1234567890",
    "michael", "654321", "superman", "1qaz2wsx", "7777777", "121212", "000000", "qazwsx",
    "123qwe", "killer", "trustno1", "jordan", "jennifer", "zxcvbnm", "asdfgh", "hunter",
    "buster", "soccer", "harley", "batman", "andrew", "tigger", "sunshine", "iloveyou",
    "2000", "charlie", "robert", "thomas", "hockey", "ranger", "daniel", "starwars", 
    "klaster", "112233", "george", "computer", "michelle", "jessica", "pepper", "1111",
    "zxcvbn", "555555", "11111111", "131313", "freedom", "777777", "pass", "maggie",
    "159753", "aaaaaa", "ginger", "princess", "joshua", "cheese", "amanda", "summer",
    "love", "ashley", "6969", "nicole", "chelsea", "matthew", "access", "yankees",
    "987654321", "dallas", "austin", "thunder", "taylor", "matrix"
])

def password_length(password):
    return len(password)

def contains_digit(password):
    return any(char.isdigit() for char in password)

def contains_special(password):
    return any(char in string.punctuation for char in password)

def contains_upper(password):
    return any(char.isupper() for char in password)

def word_match(password):
    return any(word.lower() in english_words for word in re.findall(r'[a-zA-Z]+', password))

def levenshtein_similarity(password):
    # Lower distance means more similar to a common weak password
    return min(levenshtein_distance(password, common) for common in common_passwords)

def character_entropy(password):
    if not password:
        return 0
    char_counts = Counter(password)
    length = len(password)
    entropy = -sum((count / length) * math.log2(count / length) for count in char_counts.values())
    return entropy

def contains_keyboard_pattern(password):
    patterns = ["qwerty", "asdf", "zxcv", "uiop", "ghjkl", "bnm", "1234", "6789", "12345", "123456789", "1234567890",
                "qwerty1234", "qwerty6789", "qwerty12345",
                "asdf1234", "asdf6789", "asdf12345",
                "zxcv1234", "zxcv6789", "zxcv12345",
                "uiop1234", "uiop6789", "uiop12345",
                "uiop1234", "uiop6789", "uiop12345",
                "ghjkl1234", "ghjkl6789", "ghjkl12345",
                "bnm1234", "bnm6789", "bnm12345"]
    return any(pattern in password.lower() for pattern in patterns)

def contains_repeated_chars(password):
    return any(password.count(char) > 2 for char in set(password))

def vowel_to_consonant_ratio(password):
    vowels = "aeiouAEIOU"
    v_count = 0
    c_count = 0
    for char in password:
        if char in vowels:
            v_count += 1
        elif char.isalpha():
            c_count += 1

    if c_count > 0:
        return v_count / c_count
    else:
        return 0


def extract_engineered_features(password):
    import numpy as np
    features = [
        password_length(password),
        int(contains_digit(password)),
        int(contains_special(password)),
        int(contains_upper(password)),
        int(word_match(password)),
        levenshtein_similarity(password),
        character_entropy(password),
        int(contains_keyboard_pattern(password)),
        int(contains_repeated_chars(password)),
        vowel_to_consonant_ratio(password)
    ]
    return np.array(features).reshape(1, -1)
