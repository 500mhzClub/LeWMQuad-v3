"""Event-driven exploration and return over observed exits and executed edges.

This is a symbolic runtime component, not a place/beacon/exit detector. Upstream
perception must supply qualified identities, and an executor must establish
arrival. None of those facts is inferred from image similarity, a selected
action, an intended destination, or a predicted trajectory. Unknown association
halts routing. No simulator geometry, coordinates or prepopulated graph is used.
"""
from dataclasses import dataclass,asdict
import math

from lewm.memory.directed_traversal_graph import DirectedTraversalGraph,Traversal


def identity(value):
    if not isinstance(value,str) or not value: raise ValueError('nonempty observation identity required')


def clock(value):
    if type(value) is not int or value<0: raise ValueError('nonnegative integer sensor timestamp required')


@dataclass(frozen=True)
class PlaceFix:
    observation_id: str
    timestamp_ns: int
    place_id: str | None
    stable: bool

    def __post_init__(self):
        identity(self.observation_id); clock(self.timestamp_ns)
        if self.place_id is not None: identity(self.place_id)
        if type(self.stable) is not bool: raise ValueError('explicit stability evidence required')


@dataclass(frozen=True)
class ExitObservation:
    observation_id: str
    timestamp_ns: int
    place_id: str
    exit_id: str
    bearing_body_rad: float

    def __post_init__(self):
        for value in (self.observation_id,self.place_id,self.exit_id): identity(value)
        clock(self.timestamp_ns)
        if (isinstance(self.bearing_body_rad,bool) or not isinstance(self.bearing_body_rad,(float,int))
                or not math.isfinite(self.bearing_body_rad) or abs(self.bearing_body_rad)>math.pi):
            raise ValueError('finite wrapped observed body bearing required')


class ObservedExploration:
    """Persistent observed frontiers with conservative directed-return planning.

No automatic retry of failed/ambiguous exits, no reverse-edge inference, no
merge from a MAP change. A new confirmed fix can relocalize the robot but cannot
retroactively qualify an ambiguous traversal. Every attempted exit remains in
history. Bearings are usable only at the same physical observation as the fix;
remote or stale exits demand fresh perception before execution. The supplied
maximum observation age is a runtime design parameter, not calibrated here.
"""
    def __init__(self,*,required_beacons=1,maximum_observation_age_ns=200_000_000):
        if type(required_beacons) is not int or required_beacons<1: raise ValueError('positive beacon count required')
        clock(maximum_observation_age_ns)
        if maximum_observation_age_ns==0: raise ValueError('positive observation age budget required')
        self.required_beacons=required_beacons; self.maximum_age=maximum_observation_age_ns
        self.graph=DirectedTraversalGraph(); self.home=None; self.fix=None; self.pending=None
        self.exits={}; self.beacons={}; self.attempts=[]; self._observations={}; self._actions=set()
        self._route_exit={}; self._last_time=-1

    def _accept(self,kind,event):
        key=event.observation_id; binding=(kind,event)
        if key in self._observations:
            if self._observations[key]!=binding: raise ValueError('conflicting observation identity reuse')
            return False
        if event.timestamp_ns<self._last_time: raise ValueError('out-of-order observation')
        self._observations[key]=binding; self._last_time=event.timestamp_ns; return True

    def _set_fix(self,fix):
        self.fix=fix
        if fix.place_id is not None:
            self.graph.observe_place(fix.place_id)
            if self.home is None: self.home=fix.place_id

    def observe_place(self,fix):
        if not isinstance(fix,PlaceFix): raise ValueError('typed place observation required')
        if self.pending is not None: raise ValueError('finish active traversal before committing arrival')
        if self._accept('place',fix): self._set_fix(fix)

    def observe_exit(self,event):
        if not isinstance(event,ExitObservation): raise ValueError('typed exit observation required')
        if (self.pending is not None or self.fix is None or self.fix.place_id!=event.place_id
                or self.fix.timestamp_ns!=event.timestamp_ns):
            raise ValueError('exit must be observed at the current associated physical fix')
        if self._accept('exit',event):
            key=(event.place_id,event.exit_id)
            if key not in self.exits:
                self.exits[key]={'status':'UNTRIED','target':None,'first_seen_ns':event.timestamp_ns,'observation':event}
            else: self.exits[key]['observation']=event

    def observe_beacon(self,beacon_id,observation_id,*,timestamp_ns):
        identity(beacon_id); identity(observation_id); clock(timestamp_ns)
        if (self.pending is not None or self.fix is None or self.fix.place_id is None
                or self.fix.timestamp_ns!=timestamp_ns): raise ValueError('beacon requires current associated observation')
        # Repeated detections of the same identity do not increase discovery count.
        event=PlaceFix(observation_id,timestamp_ns,self.fix.place_id,self.fix.stable)
        if self._accept('beacon:'+beacon_id,event):
            self.beacons.setdefault(beacon_id,[]).append(event)

    def begin(self,action_id,exit_id,*,now_ns):
        identity(action_id); identity(exit_id); clock(now_ns)
        if action_id in self._actions: raise ValueError('action identity already used')
        if self.pending is not None: raise ValueError('one traversal at a time')
        if self._readiness(now_ns) is not None: raise ValueError('current stable fresh associated fix required')
        key=(self.fix.place_id,exit_id)
        if key not in self.exits: raise ValueError('unobserved exit cannot be executed')
        entry=self.exits[key]
        if entry['status'] not in ('UNTRIED','VISITED'): raise ValueError('failed or uncertain exit requires separate new qualification')
        if entry['observation'].timestamp_ns!=self.fix.timestamp_ns: raise ValueError('fresh exit bearing required')
        self.pending={'action_id':action_id,'source':key[0],'exit_id':exit_id,'started_ns':now_ns,'expected_target':entry['target']}
        self._actions.add(action_id); self._last_time=now_ns

    def finish(self,action_id,fix,*,reached,viable_arrival):
        if not isinstance(fix,PlaceFix) or type(reached) is not bool or type(viable_arrival) is not bool:
            raise ValueError('typed actual arrival evidence required')
        if self.pending is None or self.pending['action_id']!=action_id: raise ValueError('no matching executed traversal')
        p=self.pending
        if fix.timestamp_ns<=p['started_ns'] or fix.observation_id in self._observations:
            raise ValueError('new positive-duration arrival observation required')
        # Validate chronology before any state mutation. A native/fault failure
        # still calls finish with its actual observed terminal time and evidence.
        if fix.timestamp_ns<self._last_time: raise ValueError('out-of-order arrival')
        target=fix.place_id; expected=p['expected_target']; entry=self.exits[(p['source'],p['exit_id'])]
        usable=bool(reached and viable_arrival and fix.stable and target is not None and target!=p['source'])
        conflict=bool(usable and expected is not None and target!=expected)
        self._accept('place',fix); self._set_fix(fix)
        duration=(fix.timestamp_ns-p['started_ns'])/1e9
        if expected is not None and (not usable or conflict):
            self.graph.record(Traversal(action_id+'-invalidate',p['source'],expected,duration,False,False,False))
        if usable and not conflict:
            self.graph.record(Traversal(action_id,p['source'],target,duration,True,True,True))
            self._route_exit[(p['source'],target)]=p['exit_id']; entry['target']=target; entry['status']='VISITED'
        else:
            entry['status']='CONFLICT' if conflict else 'UNCERTAIN' if target is None else 'FAILED'
        self.attempts.append(p|{'finished_ns':fix.timestamp_ns,'arrival_observation_id':fix.observation_id,
            'observed_target':target,'reached':reached,'viable_arrival':viable_arrival,'stable':fix.stable,
            'qualified':usable and not conflict,'association_conflict':conflict})
        self.pending=None

    def _readiness(self,now_ns):
        clock(now_ns)
        if now_ns<self._last_time: raise ValueError('decision clock predates latest event')
        if self.pending is not None: return 'EXECUTING'
        if self.fix is None or self.fix.place_id is None: return 'LOCALIZE'
        if now_ns-self.fix.timestamp_ns>self.maximum_age: return 'OBSERVE_PLACE'
        if not self.fix.stable: return 'STABILIZE'
        return None

    def _exit_decision(self,place,exit_id,reason):
        entry=self.exits[(place,exit_id)]; event=entry['observation']
        common={'place_id':place,'exit_id':exit_id,'reason':reason}
        if event.timestamp_ns!=self.fix.timestamp_ns: return {'kind':'OBSERVE_EXIT',**common}
        return {'kind':'TRAVERSE_EXIT',**common,'bearing_body_rad':event.bearing_body_rad,
            'observation_id':event.observation_id,'observation_ns':event.timestamp_ns}

    def _route_decision(self,path,reason):
        return self._exit_decision(path[0],self._route_exit[(path[0],path[1])],reason)|{'route':path}

    def decide(self,*,now_ns,return_home=False):
        if type(return_home) is not bool: raise ValueError('explicit return request required')
        readiness=self._readiness(now_ns)
        if readiness is not None: return {'kind':readiness}
        current=self.fix.place_id; returning=return_home or len(self.beacons)>=self.required_beacons
        if returning:
            path=self.graph.route(current,self.home)
            if path==[current]:
                return {'kind':'MISSION_COMPLETE' if len(self.beacons)>=self.required_beacons else 'HOME',
                    'place_id':current,'observed_beacon_count':len(self.beacons)}
            if path is not None: return self._route_decision(path,'directed_return')
        # Return without a known directed path requires more exploration, not
        # an invented inverse of the outward route. Local frontiers first.
        frontiers=sorted(((entry['first_seen_ns'],place,exit_id) for (place,exit_id),entry in self.exits.items()
            if entry['status']=='UNTRIED'))
        reason='seek_return_route' if returning else 'explore_observed_frontier'
        for _,place,exit_id in frontiers:
            if place==current: return self._exit_decision(place,exit_id,reason)
        reachable=[]
        for first_seen,place,exit_id in frontiers:
            path=self.graph.route(current,place)
            if path is not None:
                cost=sum(self.graph.edge_summary(a,b)['mean_success_duration_s'] for a,b in zip(path,path[1:]))
                reachable.append((cost,first_seen,place,exit_id,path))
        if reachable: return self._route_decision(min(reachable)[-1],reason)
        return {'kind':'NO_VERIFIED_RETURN' if returning else 'NO_REACHABLE_OBSERVED_FRONTIER',
            'observed_untried_exits':len(frontiers),'attempts':len(self.attempts)}

    def snapshot(self):
        """Serializable audit view, not an oracle graph or persistence restore API."""
        return {'home':self.home,'fix':asdict(self.fix) if self.fix is not None else None,
            'pending':dict(self.pending) if self.pending is not None else None,
            'exits':[{'place_id':p,'exit_id':e,**{k:v for k,v in row.items() if k!='observation'},
                'observation':asdict(row['observation'])} for (p,e),row in sorted(self.exits.items())],
            'beacons':{k:[asdict(e) for e in v] for k,v in self.beacons.items()},
            'attempts':[dict(a) for a in self.attempts]}
