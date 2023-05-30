"""Contains various string processing utils
"""
from typing import List, Any, Union, Tuple
import re


def soft_set(input_list: List[Any]) -> List[Any]:
    """Returns set while preserving order

    Args:
        arr (List[Any]): List

    Returns:
        List[Any]: List with unique values only with intact order
    Alternative:
        [seen.setdefault(x, x) for x in input_list if x not in seen]
        Slightly faster, less readable
    """
    return sorted(set(input_list), key=input_list.index)


def remove_emoji(text: str):
    """Remove emoji from text.
    Parameters
    ----------
    text: :str
    """
    regrex_pattern = re.compile(pattern="["
                                u"\U0001F600-\U0001F64F"
                                u"\U0001F300-\U0001F5FF"
                                u"\U0001F680-\U0001F6FF"
                                u"\U0001F1E0-\U0001F1FF"
                                u"\U00002500-\U00002BEF"
                                u"\U00002702-\U000027B0"
                                u"\U00002702-\U000027B0"
                                u"\U000024C2-\U0001F251"
                                u"\U0001f926-\U0001f937"
                                u"\U00010000-\U0010ffff"
                                u"\u2640-\u2642"
                                u"\u2600-\u2B55"
                                u"\u200d"
                                u"\u23cf"
                                u"\u23e9"
                                u"\u231a"
                                u"\ufe0f"
                                u"\u3030"
                                "]+", flags=re.UNICODE)
    return regrex_pattern.sub(r'', text)


def remove_chars(text: str, charr_arr: Union[List[str], None] = None):
    """Removes given chars from string

    Args:
        text (str):
        charr_arr (List[str], optional): _description_. Defaults to ['?', '$',
        '.', '!', '»', ')', '(', '%', '•', '”', '=', ',', '&', '<', '>', '-'].
    Returns:
        str:
    """
    if not charr_arr:
        charr_arr = ['?', '$', '.', '!',
                     '»', ')', '(', '%',
                     '•', '”', '=', ',',
                     '&', '<', '>', '-']
    return re.sub(f"[{'|'.join(charr_arr)}]", r'', text)


def get_word_trigrams(sentence: str) -> List[Tuple[str, str, str]]:
    """Divides a sentence into word trigrams.

    Args:
        sentence (str): Input sentence.

    Returns:
        List[Tuple[str, str, str]]: List of word trigrams.
    """
    words = sentence.split()
    trigrams = [(words[i], words[i+1], words[i+2]) for i in range(
        len(words) - 2)]
    return trigrams


def get_word_bigrams(sentence: str) -> List[Tuple[str, str]]:
    """Divides a sentence into word bigrams.

    Args:
        sentence (str): Input sentence.

    Returns:
        List[Tuple[str, str]]: List of word bigrams.
    """
    words = sentence.split()
    bigrams = [(words[i], words[i+1]) for i in range(len(words) - 1)]
    return bigrams
