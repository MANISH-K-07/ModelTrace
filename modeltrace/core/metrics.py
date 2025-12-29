def calculate_sparsity(stats):
    conv_s = []
    fc_s = []

    for name, sparsity in stats.items():
        if "conv" in name.lower():
            conv_s.append(sparsity)
        elif "fc" in name.lower() or "linear" in name.lower():
            fc_s.append(sparsity)

    conv_avg = sum(conv_s) / len(conv_s) if conv_s else 0.0
    fc_avg = sum(fc_s) / len(fc_s) if fc_s else 0.0

    total = (conv_avg + fc_avg) / 2
    return conv_avg, fc_avg, total
