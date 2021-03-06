"""

https://onlinelibrary.wiley.com/doi/abs/10.1111/add.14699
"""

from importlib_resources import files
from functools import lru_cache

import pandas as pd
import numpy as np


likert_files = files('likert')
itow_data_files = likert_files / 'examples' / 'in_their_own_words' / 'data'
# itow_survey = itow_data_files / 'in_their_own_words_survey_2021_11.xlsx'
itow_survey = itow_data_files / 'in_their_own_words_survey_2022_01.xlsx'


import pandas as pd

# from graze import graze
import io

# src = 'https://www.dropbox.com/s/ycdauvy63t51tgo/Wishes%20Survey%20Boeri_NOV21%2C%202021.xlsx?dl=0'
# data = pd.read_excel(io.BytesIO(graze(src)), header=0)


@lru_cache(maxsize=1)
def get_data_and_questions(data_src=itow_survey):
    data = pd.read_excel(itow_survey, header=0)
    orig_shape = data.shape
    if data.shape == orig_shape:
        questions = data.iloc[0]
        data.drop(axis=0, index=0, inplace=True)
    return data, questions


def word_cloud(
    words,
    save_filepath=None,
    width=2538,
    height=2538,
    background_color='black',
    **kwargs,
):
    from wordcloud import WordCloud
    from typing import Mapping
    from collections import Counter

    if isinstance(words, (list, tuple)):
        weight_for_word = Counter(words)
    else:
        weight_for_word = words
    wc = WordCloud(
        width=width, height=height, background_color=background_color, **kwargs
    )
    wc.fit_words(weight_for_word)
    if save_filepath is not None:
        wc.to_file(save_filepath)
    return wc


import re


def extract_end_of_large_question(x, pattern=re.compile('(?<=-\ )[\-\w\ ]+$')):
    m = pattern.search(x)
    if m is not None:
        return m.group(0)
    else:
        return x


from flair.models import TextClassifier
from flair.data import Sentence

sia = TextClassifier.load('en-sentiment')


def _sentiment_score_object(string):
    sentence = Sentence(string)
    sia.predict(sentence)
    return sentence.labels[0]


def sentiment_score(string):
    score = _sentiment_score_object(string)
    if score.value == 'NEGATIVE':
        return -score.score
    elif score.value == 'POSITIVE':
        return score.score
    else:
        raise ValueError(f"Didn't know score.value could be {score.value}")


# ---------------------------------------------------------------------------------------

from collections import Counter

term_mapping = {
    'addict': 'Addict',
    'slang': 'Slang',
    'user': 'User',
    'other': 'Other',
}

label_mapping = {
    'others who use drugs': 'Others who use',
    'drug counselors': 'Counselor',
    'family': 'Family',
    'doctors': 'Doctor',
    '12-Step mutual support members': '12-Step',
}

# @lru_cache(maxsize=1)
def get_term_category(data):
    term_category = data['Q14']
    term_category = term_category.apply(term_mapping.get)
    term_category.name = 'label'
    return term_category


def iter_pairs(category, contexts):
    for context, context_vals in contexts.items():
        for cat, context_val in zip(category, context_vals):
            if context_val is True:
                yield context, cat


# @lru_cache(maxsize=1)
def get_q2_data(data, questions):
    q2_cols = data.columns[[c.startswith('Q2') for c in data.columns]]
    q2_questions = list(map(questions.__getitem__, q2_cols))
    # print(f"{q2_cols=}")
    # print(f"{q2_questions=}")
    return q2_cols, q2_questions


# @lru_cache(maxsize=1)
def get_term_interlocutor_data(data, questions):
    q2_cols, q2_questions = get_q2_data(data, questions)
    s = data[q2_cols]
    #
    # context_labels = list(map(extract_end_of_large_question, q2_questions))
    # s.columns = context_labels
    # s = s.applymap(lambda x: {'Yes': True, 'No': False}.get(x, x))

    context_labels = list(map(extract_end_of_large_question, q2_questions))
    # Map data (automatically extracted) labels to the ones used in paper
    if set(label_mapping) == set(context_labels):
        context_labels = list(map(label_mapping.get, context_labels))
    else:
        print(f"Oops, wasn't as expected. Using raw context_labels instead.")

    # print(f"{context_labels=}")

    s = data[q2_cols]
    s.columns = context_labels
    s = s.applymap(lambda x: {'Yes': True, 'No': False}.get(x, x))

    return s


def get_multiple_counts(data, questions):
    s = get_term_interlocutor_data(data, questions)
    n = len(s)
    term_category = get_term_category(data)

    # missing = n - total_counts; missing.name = 'NA/missing'
    missing = s.isna().sum().loc[list(label_mapping.values())]
    missing.name = 'NA/missing'

    other_counts = Counter(iter_pairs(term_category, s))
    other_counts = pd.Series(other_counts).unstack().T
    other_counts = other_counts.loc[list(term_mapping.values())][
        list(label_mapping.values())
    ]
    # print(other_counts)
    term_counts = pd.Series(Counter(term_category)).loc[list(term_mapping.values())]
    term_counts.name = 'Self'

    counts = pd.concat([term_counts, other_counts], axis=1)
    # reorder index and columns
    counts = counts.loc[list(term_mapping.values())][
        ['Self'] + list(label_mapping.values())
    ]

    total_counts = counts.sum()
    total_counts.name = 'n'

    interlocutor_transparency_counts = total_counts.iloc[1:].loc[
        list(label_mapping.values())
    ]
    interlocutor_total_counts = n - missing

    all_counts = pd.concat([counts, pd.DataFrame(total_counts).T], axis=0)

    return dict(
        n=n,
        missing=missing,
        other_counts=other_counts,
        term_counts=term_counts,
        counts=counts,
        total_counts=total_counts,
        interlocutor_transparency_counts=interlocutor_transparency_counts,
        interlocutor_total_counts=interlocutor_total_counts,
        all_counts=all_counts,
    )
    # return n, missing, other_counts, term_counts, counts, total_counts,
    #        interlocutor_transparency_counts, interlocutor_total_counts, all_counts


convert_to_percentage = lambda x, n=1: round(x * 100, n)


@np.vectorize
def _elementwise_string_join(*elements, sep=' '):
    return sep.join(map(str, elements))


def elementwise_string_join(*elements, sep=' '):
    first_df, *_ = elements
    return pd.DataFrame(
        data=_elementwise_string_join(*elements, sep=sep),
        columns=first_df.columns,
        index=first_df.index,
    )


def which_axis(df, series):
    """Returns the axis matching the series' index (0=rows, 1=columns) or None if no
    match.
    Not exact, but sufficient.
    """
    series_indices = set(series.index)
    if series_indices == set(df.index):
        return 0
    elif series_indices == set(df.columns):
        return 1
    else:
        return None


def auto_divide(df, series):
    """Divides df by series, figuring out the axis all by itself"""
    return df.divide(series, axis=which_axis(df, series))


from statsmodels.stats.proportion import proportion_confint


def proportion_confint_df(
    counts, total_counts=None, method='beta', alpha=0.05, interval_chars='()',
):
    open_char, close_char = interval_chars
    if total_counts is None:
        total_counts = counts.sum()
    axis = which_axis(counts, total_counts)

    if axis == 0:
        counts = counts.T
        total_counts = total_counts.T

    #         counts, total_counts = map(pd.DataFrame.transpose, [counts, total_counts])
    def gen():
        for cat in counts.columns:
            # note: method='beta' is for the so-called "exact" method
            lo, hi = proportion_confint(
                counts[cat], total_counts[cat], method=method, alpha=alpha
            )
            lo *= 100
            hi *= 100
            series = pd.Series(
                data=list(
                    map(
                        lambda x: f'{open_char}{x[0]:.1f}-{x[1]:.1f}%{close_char}',
                        zip(lo, hi),
                    )
                ),
                index=counts.index,
            )
            series.name = cat
            yield series

    df = pd.DataFrame(gen())
    if axis == 1:
        return df.T
    else:
        return df


def proportion_and_confint(
    counts, total_counts=None, method='beta', alpha=0.05, interval_chars='()',
):
    if total_counts is None:
        total_counts = counts.sum()
    proportions_df = auto_divide(counts, total_counts)
    confint_df = proportion_confint_df(
        counts, total_counts, method=method, alpha=alpha, interval_chars=interval_chars,
    )
    return proportions_df, confint_df


def proportion_with_confint(
    counts, total_counts=None, method='beta', alpha=0.05, interval_chars='()',
):
    proportions_df, confint_df = proportion_and_confint(
        counts, total_counts, method=method, alpha=alpha, interval_chars=interval_chars,
    )
    return elementwise_string_join(
        (proportions_df * 100).round(2).applymap(lambda x: f'{x}%'), confint_df
    )


# --------------------------


def print_shortest_and_longest(names):
    unik_names = set(names)
    print('Smallest:')
    print('\t' + '\n\t'.join(sorted(unik_names, key=len, reverse=False)[:5]))
    print('\nLargest:')
    print('\t' + '\n\t'.join(sorted(unik_names, key=len, reverse=True)[:5]))
