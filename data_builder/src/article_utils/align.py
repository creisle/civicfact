import difflib
import re

import numpy as np
import pandas as pd

ASCII_MAX = 127


"""   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
193569103  919.415    0.000 1140.304    0.000 /projects/creisle_prj/git/thesis_work/src/article_utils/align.py:87(calc_best_move)
193569103   94.637    0.000   94.637    0.000 /projects/creisle_prj/git/thesis_work/src/article_utils/align.py:116(<listcomp>)
    6   81.103   13.517 1221.414  203.569 /projects/creisle_prj/git/thesis_work/src/article_utils/align.py:64(_nw_find_paths)
892895250   53.785    0.000   53.785    0.000 {method 'append' of 'list' objects}
193569682   43.354    0.000   43.355    0.000 {built-in method builtins.max}
387138206   29.123    0.000   29.123    0.000 /projects/creisle_prj/git/thesis_work/src/article_utils/align.py:69(<lambda>)
    6   22.988    3.831 1244.406  207.401 /projects/creisle_prj/git/thesis_work/src/article_utils/align.py:235(needleman_wunsch_local_match)
    1199    0.782    0.001    0.782    0.001 {built-in method io.open_code}
195/190    0.184    0.001    0.189    0.001 {built-in method _imp.create_dynamic}
    1    0.169    0.169    0.178    0.178 {method 'read_low_memory' of 'pandas._libs.parsers.TextReader' objects}
"""


def range_overlap(range1, range2):
    if any([pd.isnull(point) for point in [*range1, *range2]]):
        return False
    if range1[1] < range2[0] or range2[1] < range1[0]:
        return False
    return True


def strip_citations(text):
    """
    Remove any in-text citations from some text. This is so that the
    text can be more easily compared to the BIOC version which strips these citations by default
    """
    text = re.sub(r'\[\d+\]', '', text)
    # text = re.sub(r'\([^)]+,\s*(1[89]\d\d|20\d\d)\s*\)', '', text)
    text = re.sub(r'\(([^)]+,\s*(1[89]\d\d|20\d\d)[a-z]?)(;\s*\1)?\)', '', text)
    text = re.sub(r'\[\d+([-–]\d+)?(, \d+([-–]\d+)?)*\]', '', text)
    return text


def simplify_text(text):
    # remove non-ascii characters
    simple_text = text.lower().encode('ascii', 'ignore').decode()
    # remove whitespace
    simple_text = re.sub(r'\s+', '', simple_text)
    return simple_text


def find_all_substring_positions(ref_string, sub_string) -> list[int]:
    start = 0
    positions = []
    while start >= 0 and start < len(ref_string):
        start = ref_string.find(sub_string, start)
        if start < 0:
            break
        positions.append(start)
        start += 1  # use start += 1 to find overlapping matches
    return positions


def _map_simplified_to_original(original, target, nws_start_index) -> tuple[int, int]:
    nws_end_index = nws_start_index + len(target) - 1  # inclusive

    start_index = -1
    end_index = -1
    current_ws_count = 0
    for index, char in enumerate(original):
        if re.match(r'\s', char) or ord(char) > ASCII_MAX:
            current_ws_count += 1
            continue
        if index - current_ws_count == nws_start_index:
            start_index = index
        if index - current_ws_count == nws_end_index:
            end_index = index
            break

    if start_index < 0 or end_index < 0:
        raise ValueError(
            f'unable to find string position ({start_index}, {end_index}) in text from mws coordinates ({nws_start_index}-{nws_end_index})'
        )

    return start_index, end_index + 1


def find_in_text(text_input, target_string_input) -> tuple[int, int]:
    """
    Ignore white-space and non-ascii differences and casing. returns the first highest scoring match
    """
    text = simplify_text(text_input)
    target = simplify_text(
        strip_citations(target_string_input)
    )  # TODO: fix normalizing to keep symbol in output

    # align the match back to the original text input string
    matches = []
    for nws_start in find_all_substring_positions(text, target):
        m_start, m_end = _map_simplified_to_original(text_input, target, nws_start)
        m_text = text_input[m_start:m_end]
        score = difflib.SequenceMatcher(None, target_string_input, m_text).ratio()

        boundary_score = 0
        if m_start > 0 and not re.match(r'(\W|-)', text_input[m_start - 1]):
            boundary_score += 1
        if m_end < len(text_input) and not re.match(r'(\W|-)', text_input[m_end]):
            boundary_score += 1

        matches.append((m_start, m_end, m_text, score, boundary_score))

    if not matches:
        raise ValueError(f'unable to find string {target} in text')
    matches = sorted(matches, key=lambda m: (m[-2] * -1, m[-1], m[0]))

    return matches[0][0], matches[0][1]


def _nw_find_paths(
    ref_seq: list,
    target_seq: list,
    gap_penalty=-1,
    ref_gap_penalty=None,
    tgt_gap_penalty=None,
    mismatch_penalty=-1,
    match_penalty=1,
    gap_exception=lambda x: False,
    open_gap_penalty=-1,
    single=True,
) -> tuple[float, list]:
    if ref_gap_penalty is None:
        ref_gap_penalty = gap_penalty
    if tgt_gap_penalty is None:
        tgt_gap_penalty = gap_penalty

    # initialize the matrix
    nrows = len(ref_seq) + 1
    ncols = len(target_seq) + 1
    matrix = np.zeros((nrows, ncols))
    paths_map: dict[tuple[int, int], list[tuple[int, int]]] = {}
    # local alignments so do not penalize end gaps (can skip initializing first row/col)
    # still need to do for paths_map
    for row_i in range(0, nrows):
        paths_map[(row_i, 0)] = [] if row_i == 0 else [(row_i - 1, 0)]
    for col_i in range(0, ncols):
        paths_map[(0, col_i)] = [] if col_i == 0 else [(0, col_i - 1)]

    def calc_best_move(row_i, col_i) -> int:
        char1 = ref_seq[row_i - 1]
        char2 = target_seq[col_i - 1]

        # from diagonal
        scores: list[tuple[int, int]] = []
        if char1 == char2:
            # match so no penalty
            scores.append(
                (matrix[row_i - 1, col_i - 1] + len(char1) * match_penalty, (row_i - 1, col_i - 1))
            )
        else:
            scores.append(
                (
                    matrix[row_i - 1, col_i - 1] + max([len(char1), len(char2)]) * mismatch_penalty,
                    (row_i - 1, col_i - 1),
                )
            )
        # from top, gap in reference
        top_score = matrix[row_i - 1, col_i]
        if not gap_exception(char1) and col_i < ncols:
            top_score += ref_gap_penalty * len(char1)
        if (
            not paths_map[(row_i - 1, col_i)]
            or (row_i - 2, col_i) != paths_map[(row_i - 1, col_i)][-1]
        ):
            # opening a new gap
            top_score += open_gap_penalty
        scores.append((top_score, (row_i - 1, col_i)))

        # from left, gap in target
        left_score = matrix[row_i, col_i - 1]
        if not gap_exception(char2) and row_i < nrows:
            left_score += tgt_gap_penalty * len(char2)
        if (
            not paths_map[(row_i, col_i - 1)]
            or (row_i, col_i - 2) != paths_map[(row_i, col_i - 1)][-1]
        ):
            # opening a new gap
            left_score += open_gap_penalty
        scores.append((left_score, (row_i, col_i - 1)))

        best_score = max([score for score, source in scores])
        paths_map[(row_i, col_i)] = []
        # now mark the paths
        for score, source in scores:
            if score == best_score:
                paths_map[(row_i, col_i)].append(source)
        return best_score

    for row_i in range(1, nrows):
        for col_i in range(1, ncols):
            matrix[row_i, col_i] = calc_best_move(row_i, col_i)

    # find the best 'paths'
    last_col = ncols - 1
    max_score = np.max(matrix[:, last_col])
    # find all starting points
    row_starts = np.where(matrix[:, last_col] == max_score)
    paths = [[(r, last_col)] for r in row_starts[0]]
    if single:
        paths = paths[:1]

    complete_paths = []

    while paths:
        path = paths.pop(0)
        tail_node = path[-1]
        choices = paths_map[tail_node]
        if single:
            choices = choices[:1]
        if not choices or tail_node[1] == 0:  # ignore ref end-gaps
            # path is complete
            complete_paths.append(path)
        else:
            for next_choice in choices:
                if tail_node[1] == ncols - 1 and next_choice[1] == ncols - 1:
                    # ignore ref end-gaps
                    paths.append(path[:-1] + [next_choice])
                else:
                    paths.append(path + [next_choice])

    # flip all the paths so they are going forward
    complete_paths = [p[::-1] for p in complete_paths]
    return max_score.item(), complete_paths


def extract_gapped_sequences(ref_seq: list[str], target_seq: list[str], path):
    ref_gaps = [[]]
    target_gaps = [[]]

    for i in range(1, len(path)):
        ref_step = path[i][0] - path[i - 1][0]
        target_step = path[i][1] - path[i - 1][1]
        if ref_step and target_step:
            # reset gap counters
            if ref_gaps[-1]:
                ref_gaps.append([])
            if target_gaps[-1]:
                target_gaps.append([])
        elif not ref_step:
            if ref_gaps[-1]:
                ref_gaps.append([])
            target_gaps[-1].append(target_seq[path[i][1] - 1])
        elif not target_step:
            if target_gaps[-1]:
                target_gaps.append([])
            ref_gaps[-1].append(ref_seq[path[i][0] - 1])

    ref_gaps = [''.join(seq) for seq in ref_gaps]
    target_gaps = [''.join(seq) for seq in target_gaps]
    return ref_gaps, target_gaps


def looks_like_hypothesis_artifact(input_string):
    """
    ex. '\u200band2B,', '\u200b(Figure2A'
    """
    input_string = re.sub(r'(Figure|and|\(|\)|\u200b|,|\d[A-Z])', '', input_string)
    return not input_string


def looks_like_citation(input_string):
    if re.match(r'\[\d+([-–]\d+)?(, \d+([-–]\d+)?)*\]', input_string):
        return True
    return False


def rescore_path(ref_seq, target_seq, path, gap_penalty, score):
    ref_gaps, target_gaps = extract_gapped_sequences(ref_seq, target_seq, path)

    ref_adjustment = 0

    for gap in ref_gaps:
        # discount terminal whitespace
        shift = len(gap) - len(gap.strip())
        ref_adjustment += abs(gap_penalty) * shift

        gap = gap.strip()

        if gap == 'deg':  # might have been using the symbol online
            ref_adjustment += abs(gap_penalty) * len(gap) + 1

    adjusted_score = score + ref_adjustment
    target_adjustment = 0

    for gap in target_gaps:
        # discount terminal whitespace
        shift = len(gap) - len(gap.strip())
        target_adjustment += abs(gap_penalty) * shift

        gap = gap.strip()
        if looks_like_citation(gap) or looks_like_hypothesis_artifact(gap):
            target_adjustment += abs(gap_penalty) * len(gap) + 1

    adjusted_score = adjusted_score + target_adjustment
    adjusted_score /= sum([len(s) for s in target_seq])

    return adjusted_score


def draw_path_alignment(sequences: "list[str]", path: "list[tuple[int,int]]", gap_character='~'):
    """
    only for debugging. show what the alignment looks like
    """
    rows = [[] for _ in sequences]

    for seq_pos in range(len(sequences)):
        for step_i in range(1, len(path)):
            curr_pos = path[step_i][seq_pos]
            prev_pos = path[step_i - 1][seq_pos]
            if curr_pos == prev_pos:
                rows[seq_pos].append(gap_character)
            else:
                rows[seq_pos].append(sequences[seq_pos][curr_pos - 1])

    rows = [''.join(row) for row in rows]
    for row in rows:
        print('> ', repr(row))


def needleman_wunsch_local_match(
    ref_seq: list, target_seq: list, **kwargs
) -> tuple[int, int, float]:
    max_score, complete_paths = _nw_find_paths(ref_seq, target_seq, **kwargs)

    # ignore things that are gapped due to citations or selection artifacts in the scoring
    paths_by_score = {}
    for path in complete_paths:
        score = rescore_path(
            ref_seq, target_seq, path, gap_penalty=kwargs.get('gap_penalty', -1), score=max_score
        )
        paths_by_score.setdefault(score, []).append(path)

    max_score = max(paths_by_score.keys())
    complete_paths = paths_by_score[max_score]

    all_positions = set()
    for path in complete_paths:
        all_positions.update([pos[0] for pos in path])

    ref_start = min(all_positions)
    ref_end = max(all_positions)

    if ref_start == ref_end:
        raise ValueError(
            f'start_index cannot be the same as the end_index. failed to align sequences with score: {max_score}'
        )
    return int(ref_start), int(ref_end), max_score


def basic_word_split(seq):
    return [c for c in re.split(r'([\s\n]+|[^a-z0-9A-Z])', seq) if c]


def word_align(ref_seq, target_seq, **kwargs):
    ref_seq = basic_word_split(ref_seq)
    target_seq = basic_word_split(target_seq)
    start_word_index, end_word_index, word_score = needleman_wunsch_local_match(
        ref_seq, target_seq, **kwargs
    )
    start_char_index = sum([len(word) for word in ref_seq[:start_word_index]])
    end_char_index = sum([len(word) for word in ref_seq[:end_word_index]])
    return start_char_index, end_char_index, word_score


def targeted_local_match(ref_seq, target_seq, kmer_size=5, max_kmer_repeat=3, **kwargs):
    # find anchor points to test more refined matching
    kmer_matches = set()
    merge_distance = len(target_seq)

    for kmer_start in range(len(target_seq) - kmer_size):
        kmer = target_seq[kmer_start : kmer_start + kmer_size]
        matches = list(re.finditer(re.escape(kmer), ref_seq))
        if len(matches) > max_kmer_repeat:
            continue
        for match in matches:
            kmer_matches.add(
                (
                    max([match.start() - merge_distance, 0]),
                    max([match.end() + merge_distance, len(ref_seq)]),
                )
            )

    merged = []

    for start, end in sorted(kmer_matches):
        if not merged or not range_overlap((start, end), merged[-1]):
            merged.append((start, end))
        else:
            merged.append((min([start, merged[-1][0]]), max([end, merged[-1][-1]])))

    alignments = []

    for ref_start, ref_end in merged:
        alignments.append(word_align(ref_seq[ref_start:ref_end], target_seq, **kwargs))

    best_alignment = max(alignments, key=lambda aln: aln[2])
    return best_alignment


def diff_strings(ref_str, input_str):
    input_i = 0
    ref_i = 0
    insertions = []
    deletions = []

    while input_i < len(input_str) or ref_i < len(ref_str):
        if ref_str[ref_i] == input_str[input_i]:
            input_i += 1
            ref_i += 1
