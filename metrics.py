def compute_metrics(text, summary, num_chunks, times):
    """
    Compute summarization pipeline metrics
    """

    char_count = len(text)

    if char_count > 0:
        compression_ratio = round((len(summary) / char_count) * 100, 2)
    else:
        compression_ratio = 0

    return {
        "total_time_sec": round(times["total"], 2),
        "extraction_time_sec": round(times["extraction"], 2),
        "summary_time_sec": round(times["summary"], 2),
        "map_time_sec": round(times["map"], 2),
        "reduce_time_sec": round(times["reduce"], 2),
        "num_chunks": num_chunks,
        "char_count": char_count,
        "compression_ratio": compression_ratio
    }