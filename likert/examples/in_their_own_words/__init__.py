"""

https://onlinelibrary.wiley.com/doi/abs/10.1111/add.14699
"""

from importlib_resources import files
from functools import lru_cache, partial
from typing import Callable, Union, Tuple, Mapping, Iterable, Optional
import pandas as pd
import numpy as np


likert_files = files("likert")
itow_files = likert_files / "examples" / "in_their_own_words"
itow_data_files = itow_files / "data"
# itow_survey = itow_data_files / 'in_their_own_words_survey_2021_11.xlsx'
itow_survey = itow_data_files / "in_their_own_words_survey_2022_01.xlsx"


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
    background_color="black",
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


def extract_end_of_large_question(x, pattern=re.compile("(?<=-\ )[\-\w\ ]+$")):
    m = pattern.search(x)
    if m is not None:
        return m.group(0)
    else:
        return x


try:
    from flair.models import TextClassifier
    from flair.data import Sentence
except (ImportError, ModuleNotFoundError) as e:
    print(f"Couldn't import flair: {e}")


@lru_cache
def get_text_classifier(data_name="en-sentiment"):
    return TextClassifier.load(data_name)


def _sentiment_score_object(string):
    sentence = Sentence(string)
    get_text_classifier().predict(sentence)
    return sentence.labels[0]


def sentiment_score(string):
    score = _sentiment_score_object(string)
    if score.value == "NEGATIVE":
        return -score.score
    elif score.value == "POSITIVE":
        return score.score
    else:
        raise ValueError(f"Didn't know score.value could be {score.value}")


def _edited_word_count_pairs(counts, word_mapping=()):
    word_mapping = dict(word_mapping)
    for word, count in counts:
        if word in word_mapping:
            word = word_mapping[word]
        yield word, count


def edit_word_counts(counts, word_mapping=()):
    """Map words of (word, count) pairs, and recompute counts

    >>> edit_word_counts([('apple', 5), ('banana', 4), ('ball', 3)], {'ball': 'banana'})
    [('banana', 7), ('apple', 5)]

    """
    c = Counter()
    for word, count in _edited_word_count_pairs(counts, word_mapping):
        c.update({word: count})
    return c.most_common()


def extract_dict(df: pd.DataFrame, key_col: str, val_col: str):
    """Extracts a dict from a dataframe

    >>> df = pd.DataFrame({'a': [1, 2, 3], 'b': ['X', 'Y', 'Z']})
    >>> extract_dict(df, 'a', 'b')
    {1: 'Z', 2: 'Y', 3: 'Z'}

    """
    return dict(zip(df[key_col], df[val_col]))


from colour import Color

ColorSpec = Union[Color, str]
Hex = str
HexGradients = Iterable[Hex]


def gradients_of_hex_colors(
    color: ColorSpec = Color("grey"), number_of_gradients=200
) -> HexGradients:
    color = Color(color)
    return list(map(str, color.range_to(Color("white"), number_of_gradients + 2)))[1:-1]


def _range_index(x, n_indices, min_x, max_x):
    return int((n_indices - 1) * (x - min_x) / (max_x - min_x))


def mk_score_to_color(
    colors: Optional[Union[ColorSpec, HexGradients]] = None,
    *,
    min_score=-1,
    max_score=1,
):
    """Makes a function that produces a color for a given score (number)"""
    if colors is None:
        colors = "grey"
    if isinstance(colors, (str, Color)):
        colors = gradients_of_hex_colors(colors)
    score_to_index = partial(
        _range_index,
        n_indices=len(colors),
        min_x=min_score,
        max_x=max_score,
    )
    return lambda score: colors[score_to_index(score)]


WordCountPair = Tuple[str, float]
CountsSpec = Union[Mapping[str, float], Iterable[WordCountPair]]
Word = str
ColorStr = str


def word_count_for_counts_with_color_control(
    counts: CountsSpec,
    word_to_score: Callable[[Word], float],
    score_to_color: Callable[[float], ColorStr],
    *,
    random_state=0,
):
    """Makes a WordCount object given

    :param counts: word count dict or pairs
    :param word_to_score: A function that scores words (gives us a number for a word)
    :param score_to_color: A function that colors scores (gives us a color for a number)
    :param random_state:
    :return:

    ``counts`` (word count dict or pairs)
    """

    def grey_color_func(
        word, font_size, position, orientation, random_state=random_state, **kwargs
    ):
        score = word_to_score(word)
        return score_to_color(score)

    wc = word_cloud(dict(counts))
    wc = wc.recolor(color_func=grey_color_func, random_state=random_state)
    return wc


# ---------------------------------------------------------------------------------------

from collections import Counter

term_mapping = {
    "addict": "Addict",
    "slang": "Slang",
    "user": "User",
    "other": "Other",
}

label_mapping = {
    "others who use drugs": "Others who use",
    "drug counselors": "Counselor",
    "family": "Family",
    "doctors": "Doctor",
    "12-Step mutual support members": "12-Step",
}


# @lru_cache(maxsize=1)
def get_term_category(data):
    term_category = data["Q14"]
    term_category = term_category.apply(term_mapping.get)
    term_category.name = "label"
    return term_category


def iter_pairs(category, contexts):
    for context, context_vals in contexts.items():
        for cat, context_val in zip(category, context_vals):
            if context_val is True:
                yield context, cat


# @lru_cache(maxsize=1)
def get_q2_data(data, questions):
    q2_cols = data.columns[[c.startswith("Q2") for c in data.columns]]
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
    s = s.applymap(lambda x: {"Yes": True, "No": False}.get(x, x))

    return s


def get_multiple_counts(data, questions):
    s = get_term_interlocutor_data(data, questions)
    n = len(s)
    term_category = get_term_category(data)

    # missing = n - total_counts; missing.name = 'NA/missing'
    missing = s.isna().sum().loc[list(label_mapping.values())]
    missing.name = "NA/missing"

    other_counts = Counter(iter_pairs(term_category, s))
    other_counts = pd.Series(other_counts).unstack().T
    other_counts = other_counts.loc[list(term_mapping.values())][
        list(label_mapping.values())
    ]
    # print(other_counts)
    term_counts = pd.Series(Counter(term_category)).loc[list(term_mapping.values())]
    term_counts.name = "Self"

    counts = pd.concat([term_counts, other_counts], axis=1)
    # reorder index and columns
    counts = counts.loc[list(term_mapping.values())][
        ["Self"] + list(label_mapping.values())
    ]

    total_counts = counts.sum()
    total_counts.name = "n"

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
def _elementwise_string_join(*elements, sep=" "):
    return sep.join(map(str, elements))


def elementwise_string_join(*elements, sep=" "):
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
    counts,
    total_counts=None,
    method="beta",
    alpha=0.05,
    interval_chars="()",
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
                        lambda x: f"{open_char}{x[0]:.1f}-{x[1]:.1f}%{close_char}",
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
    counts,
    total_counts=None,
    method="beta",
    alpha=0.05,
    interval_chars="()",
):
    if total_counts is None:
        total_counts = counts.sum()
    proportions_df = auto_divide(counts, total_counts)
    confint_df = proportion_confint_df(
        counts,
        total_counts,
        method=method,
        alpha=alpha,
        interval_chars=interval_chars,
    )
    return proportions_df, confint_df


def proportion_with_confint(
    counts,
    total_counts=None,
    method="beta",
    alpha=0.05,
    interval_chars="()",
):
    proportions_df, confint_df = proportion_and_confint(
        counts,
        total_counts,
        method=method,
        alpha=alpha,
        interval_chars=interval_chars,
    )
    return elementwise_string_join(
        (proportions_df * 100).round(2).applymap(lambda x: f"{x}%"), confint_df
    )


def compute_stats_on_the_difference_of_proportions(
    p1, p2, n1, n2, alternative="two-sided", prop_var=False
):
    r"""Compute the test statistic and p-value for the difference of two proportions.

    :param p1: proportion in the first sample
    :param p2: proportion in the second sample
    :param n1: sample size of the first sample
    :param n2: sample size of the second sample
    :param alternative: alternative hypothesis, either 'two-sided' (default), 'smaller' or 'larger'
    :param prop_var: if False (default), perform a normal approximation (agrees with R's prop.test).
                        if True, compute the exact p-value based on the proportion variance.
    :return: a tuple containing the test statistic and the p-value

    Examples
    --------

    >>> p_value, chi_squared, z_score = compute_stats_on_the_difference_of_proportions(
    ...     0.1, 0.15, 100, 100
    ... )
    >>> print(
    ...     f"p-value: {p_value:.3f}",
    ...     f"chi-squared: {chi_squared:.3f}",
    ...     f"z-score: {z_score:.3f}",
    ...     sep='\n'
    ... )
    p-value: 0.285
    chi-squared: 1.143
    z-score: -1.069

    """
    from statsmodels.stats.proportion import proportions_ztest, proportions_chisquare

    if all(p > 1 for p in [p1, p2]):
        # assume p1 and p2 are NOT proportions BUT counts
        # so convert to proportions by dividing by n1 and n2
        p1 /= n1
        p2 /= n2

    proportions = np.array([p1, p2])
    sample_sizes = np.array([n1, n2])
    counts = proportions * sample_sizes

    # perform z-test
    z_score, p_value = proportions_ztest(
        count=counts, nobs=sample_sizes, alternative=alternative, prop_var=prop_var
    )
    # perform chi-square test
    chi_squared, p_value, table = proportions_chisquare(count=counts, nobs=sample_sizes)

    return p_value, chi_squared, z_score


def compute_stats_on_the_difference_of_proportions_df(
    df: pd.DataFrame,
    p1_col: str = "p1",
    p2_col: str = "p2",
    n1_col: str = "n1",
    n2_col: str = "n2",
    alternative="two-sided",
    prop_var=False,
):
    """Computes the multiple difference of proportions stats given a dataframe with
    the two proportions (or counts) and their sample sizes in specific columns.

    >>> df = pd.DataFrame({
    ...     'p1': [0.1, 0.15],
    ...     'p2': [0.15, 0.2],
    ...     'n1': [100, 100],
    ...     'n2': [100, 100],
    ... })
    >>> compute_stats_on_the_difference_of_proportions_df(df)
        p_value  chi_squared   z_score
    0  0.285049     1.142857 -1.069045
    1  0.352120     0.865801 -0.930484

    """
    # use compute_stats_on_the_difference_of_proportions to compute the stats
    # for each row

    stats = df.apply(
        lambda row: compute_stats_on_the_difference_of_proportions(
            row[p1_col],
            row[p2_col],
            row[n1_col],
            row[n2_col],
            alternative=alternative,
            prop_var=prop_var,
        ),
        axis=1,
    )
    # unpack the stats into separate columns
    stats_df = pd.DataFrame(
        data=list(stats),
        columns=["p_value", "chi_squared", "z_score"],
        index=df.index,
    )
    return stats_df


# --------------------------


def print_shortest_and_longest(names):
    unik_names = set(names)
    print("Smallest:")
    print("\t" + "\n\t".join(sorted(unik_names, key=len, reverse=False)[:5]))
    print("\nLargest:")
    print("\t" + "\n\t".join(sorted(unik_names, key=len, reverse=True)[:5]))
