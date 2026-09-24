"""Validate pinned py-spy data and summarize only marked controller stacks.

Pure parsing: caller authenticates files, tool/child identities and ended owners.
Weights are nominal sample weights, not measured CPU self time or wall duration.
"""
from collections import Counter
import math
from scripts.external_body_projection_profile_replay_development import MARKERS, MARKER_WINDOWS, WINDOWS


def number(value):
    return type(value) in (float,int) and math.isfinite(value)


def summarize(profile, *, marker_source_file, owner_native_thread_id):
    if (type(owner_native_thread_id) is not int or owner_native_thread_id <= 0
            or type(marker_source_file) is not str or not marker_source_file.startswith('/')):
        raise ValueError('exact original main-thread id and absolute marker source required')
    if (type(profile) is not dict or profile.get('exporter') != 'py-spy@0.4.2'
            or profile.get('$schema') != 'https://www.speedscope.app/file-format-schema.json'):
        raise ValueError('pinned external sampled-profile format required')
    frames=profile['shared']['frames'];profiles=profile['profiles']
    if (type(frames) is not list or not 1 <= len(frames) <= 100_000
            or type(profiles) is not list or not 1 <= len(profiles) <= 128):
        raise ValueError('bounded nonempty frames and thread profiles required')
    locations=[]
    for frame in frames:
        if (type(frame) is not dict or set(frame) != {'name','file','line','col'}
                or type(frame['name']) is not str or type(frame['file']) is not str
                or type(frame['line']) is not int or frame['line'] < 0
                or (frame['col'] is not None and (type(frame['col']) is not int or frame['col'] < 0))):
            raise ValueError('exact Python stack-frame metadata required')
        if any(part in ('sealed','sealed_test.json') or part.startswith('sealed_')
                for part in frame['file'].replace('\\','/').split('/')):
            raise ValueError('protected frame cannot be summarized')
        locations.append((frame['file'],frame['line'],frame['name']))
    reverse={name:frame for frame,name in MARKERS.items()}
    observed={frame:dict(samples=0,controller_samples=0,nominal_sample_weight_s=0.) for frame in MARKERS}
    counts={window:dict(leaf=Counter(),inclusive_functions=Counter(),inclusive_files=Counter()) for window in WINDOWS}
    total=unmarked=wrapper_only=0;owner_names=set()
    for thread in profiles:
        if (thread.get('type') != 'sampled' or thread.get('unit') != 'seconds'
                or type(thread.get('name')) is not str
                or not number(thread.get('startValue')) or thread['startValue'] != 0
                or not number(thread.get('endValue')) or thread['endValue'] < 0):
            raise ValueError('finite sampled thread timeline in seconds required')
        samples=thread['samples'];weights=thread['weights']
        if (type(samples) is not list or type(weights) is not list or len(samples)!=len(weights)
                or total+len(samples)>1_000_000
                or any(not number(weight) or weight!=.01 for weight in weights)
                or not math.isclose(sum(weights),thread['endValue'],rel_tol=0,abs_tol=1e-8)):
            raise ValueError('bounded complete 100-Hz nominal sample population required')
        for sample,weight in zip(samples,weights,strict=True):
            if (type(sample) is not list or not 1 <= len(sample) <= 256
                    or any(type(index) is not int or not 0 <= index < len(locations) for index in sample)):
                raise ValueError('bounded original stack indices required')
            total+=1;markers=[]
            for position,index in enumerate(sample):
                file,_,name=locations[index]
                if file==marker_source_file and name.startswith('controller_observation_'):
                    if name not in reverse:raise ValueError('unexpected controller observation marker')
                    markers.append((position,reverse[name]))
            if not markers:
                unmarked+=1;continue
            if len(markers)!=1 or not thread['name'].startswith(f'Thread {owner_native_thread_id} '):
                raise ValueError('one original owner observation marker per sample required')
            owner_names.add(thread['name'])
            position,frame=markers[0];row=observed[frame]
            row['samples']+=1;row['nominal_sample_weight_s']+=weight
            inside=[locations[index] for index in sample[position+1:]]
            if not inside:
                wrapper_only+=1;continue
            row['controller_samples']+=1;window=counts[MARKER_WINDOWS[frame]]
            window['leaf'][inside[-1]]+=1
            # Function recursion and multiple source lines count once per sample.
            for key in set((file,name) for file,_,name in inside):
                window['inclusive_functions'][key]+=1
            for file in set(item[0] for item in inside):window['inclusive_files'][file]+=1
    if len(owner_names)!=1 or any(row['controller_samples']==0 for row in observed.values()):
        raise ValueError('all thirty original marked observations require controller samples')
    summaries={}
    for name in WINDOWS:
        observations=[dict(frame=frame,**observed[frame]) for frame in MARKERS if MARKER_WINDOWS[frame]==name]
        counters=counts[name]
        summaries[name]=dict(observations=observations,
            sampled_leaf_locations=[dict(file=file,line=line,function=function,samples=count)
                for (file,line,function),count in sorted(counters['leaf'].items(),key=lambda pair:(-pair[1],pair[0]))],
            inclusive_function_samples=[dict(file=file,function=function,samples=count)
                for (file,function),count in sorted(counters['inclusive_functions'].items(),key=lambda pair:(-pair[1],pair[0]))],
            inclusive_file_samples=[dict(file=file,samples=count)
                for file,count in sorted(counters['inclusive_files'].items(),key=lambda pair:(-pair[1],pair[0]))])
    return dict(schema='body_projection_external_controller_samples_development.v1',
        exporter=profile['exporter'],owner_profile_name=next(iter(owner_names)),
        total_samples=total,unmarked_samples=unmarked,marked_samples=total-unmarked,
        wrapper_only_samples=wrapper_only,marked_observations=30,windows=summaries,
        all_marked_observations_sampled=True,unmarked_input_admission_excluded_from_controller_summary=True,
        nominal_sample_weights_are_measured_durations=False,cpu_self_time_established=False,
        sampling_bias_excluded=False,profiler_overhead_removed=False,
        native_execution=False,navigation_qualified=False,real_time_qualified=False)
