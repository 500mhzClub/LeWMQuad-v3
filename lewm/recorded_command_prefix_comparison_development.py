"""Bound shadow-command comparisons by actual executed command differences."""


def compare(reference,shadow,executed_commands):
    if not reference or len(reference)!=len(shadow) or len(executed_commands)!=len(shadow)-1:
        raise ValueError('complete observation prefix and its executed intervals required')
    if any(len(c)!=3 for c in executed_commands):raise ValueError('three-axis executed command required')
    if any(r['requested_command']!=c for r,c in zip(reference,executed_commands)):
        raise ValueError('reference decisions must match the actual recorded command prefix')
    first=next((i for i,(r,c) in enumerate(zip(shadow,executed_commands)) if r['requested_command']!=c),None)
    terminal=next((i for i,(a,b) in enumerate(zip(reference,shadow,strict=True)) if a['terminal']!=b['terminal']),None)
    limits=[i for i in (first,terminal) if i is not None]
    frames=min(len(shadow),min(limits)+1) if limits else len(shadow)
    return dict(observations=len(shadow),executed_intervals=len(executed_commands),
        first_executed_command_difference=first,first_terminal_difference=terminal,
        common_executed_prefix_observations=frames,
        includes_observation_before_first_command_difference=True,
        last_observation_command_was_executed=False,
        last_observation_proposal_matches=reference[-1]['requested_command']==shadow[-1]['requested_command'],
        all_recorded_commands_match=first is None,all_terminal_states_match=terminal is None,
        prospective_closed_loop=False,unexecuted_outcomes_inferred=False)
