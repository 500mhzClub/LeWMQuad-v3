"""Reference ingestion coverage for native terminal stops on a command clock."""


def expected_ingested(camera,tape,*,prefix_end_time,terminal_index,stop_reason):
    expected=[i for i,row in enumerate(camera) if row['timestamp_s']<=prefix_end_time]
    lookup={row['physical_sample_index']:i for i,row in enumerate(camera)}
    if len(lookup)!=len(camera): raise ValueError('duplicate camera boundary')
    for entry in tape:
        if entry['stage']!='control' or entry['post_sample_index']-entry['pre_sample_index']!=50: continue
        end=entry['post_sample_index']
        # _sample records the native stopping sample, then raises PhysicalStop.
        # Even if it is the 50th sample, command_tick never returns and the
        # caller's post-command observe callback is not reached. Final image
        # capture is retained for evaluation, not ingested by the halted policy.
        if end==terminal_index and stop_reason in ('DISALLOWED_CONTACT','BODY_STABILITY_LIMIT'): continue
        if end not in lookup: raise ValueError('missing completed nonterminal command image')
        expected.append(lookup[end])
    return sorted(set(expected))
