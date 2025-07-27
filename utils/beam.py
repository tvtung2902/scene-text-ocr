import math

def log_sum_exp(a, b):
    if a == -float('inf'):
        return b
    if b == -float('inf'):
        return a
    return max(a, b) + math.log1p(math.exp(-abs(a - b)))

def beam_search_decode(log_probs, beam_width=5, blank_index=1):
    T, B, C = log_probs.size()
    log_probs = log_probs.cpu()
    final_results = []

    for b in range(B):
        beams = [([], 0.0)]
        for t in range(T):
            new_beams = {}
            for prefix, score in beams:
                for c in range(C):
                    p = log_probs[t, b, c].item()
                    new_prefix = prefix + [c]

                    if len(prefix) > 0 and c == prefix[-1]:
                        if c == blank_index:
                            key = tuple(prefix)
                            new_beams[key] = log_sum_exp(new_beams.get(key, -float('inf')), score + p)
                        else:
                            key = tuple(new_prefix)
                            new_beams[key] = log_sum_exp(new_beams.get(key, -float('inf')), score + p)
                    else:
                        key = tuple(new_prefix)
                        new_beams[key] = log_sum_exp(new_beams.get(key, -float('inf')), score + p)

            beams = sorted(new_beams.items(), key=lambda x: x[1], reverse=True)[:beam_width]
            beams = [(list(k), v) for k, v in beams]

        best_seq = beams[0][0]
        final_results.append(best_seq)

    return final_results

