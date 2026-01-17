import re
import math

def last_boxed_only_string(string):
    idx = string.rfind('\\boxed')
    if idx < 0:
        idx = string.rfind('\\fbox')
        if idx < 0:
            return None

    i = idx
    right_brace_idx = None
    num_left_braces_open = 0
    while i < len(string):
        if string[i] == '{':
            num_left_braces_open += 1
        if string[i] == '}':
            num_left_braces_open -= 1
            if num_left_braces_open == 0:
                right_brace_idx = i
                break
        i += 1

    if right_brace_idx is None:
        retval = None
    else:
        retval = string[idx:right_brace_idx + 1]

    return retval


def remove_boxed(s):
    left = '\\boxed{'
    try:
        assert s[:len(left)] == left
        assert s[-1] == '}'
        return s[len(left):-1]
    except Exception:
        return None


def extract_boxed_answer(pred_str, strip_double_curly_brace=False):
    boxed_str = last_boxed_only_string(pred_str)
    if boxed_str is None:
        return ""
    answer = remove_boxed(boxed_str)
    if answer is None:
        return ""
    if strip_double_curly_brace:
        match = re.match('^\{(.*)\}$', answer)  # noqa: W605
        if match:
            answer = match.group(1)
    if answer:
        return answer
    else:
        return ""
    
def normalize_latex_text(s):
    if not s:
        return ""

    s = re.sub(r"\\text\s*\{(.*?)\}", r"\1", s)

    s = s.replace("\\ ", " ")
    s = s.replace("\\", "")

    s = s.strip()

    s = re.sub(r"\s+", " ", s)
    if s: return s
    else: return ""

def normalized_entropy(counts):
    values = list(counts.values())
    total = sum(values)

    if total == 0 or len(values) <= 1:
        return 0.0

    probs = [v / total for v in values]
    H = -sum(p * math.log(p + 1e-12) for p in probs)

    N = len(values)
    H_max = math.log(N)

    return H / H_max
