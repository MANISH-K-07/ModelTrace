def calculate_sparsity(stats):
    conv_s = []
    fc_s = []

    for name, sparsity in stats.items():
        lname = name.lower()
        if "conv" in lname:
            conv_s.append(sparsity)
        elif "fc" in lname or "linear" in lname:
            fc_s.append(sparsity)

    conv_avg = sum(conv_s) / len(conv_s) if conv_s else 0.0
    fc_avg = sum(fc_s) / len(fc_s) if fc_s else 0.0

    total = sum(stats.values()) / len(stats) if stats else 0.0
    return conv_avg, fc_avg, total
