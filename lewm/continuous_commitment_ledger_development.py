"""Known future commands and actual-prefix checks for overlapping planning.

Each plan commits a 400-ms command after a 300-ms known prefix. A veto during
that prefix invalidates the next plan; it cannot execute forecasts conditioned
on commands the actuator was never asked to perform.
"""
from dataclasses import dataclass

PERIOD=100_000_000
SERVICE=20_000_000


@dataclass(frozen=True)
class Commitment:
    plan: object
    completed_ns: int
    prefix: tuple


class ContinuousCommitmentLedger:
    def __init__(self):
        self.commitments=[];self.vetoes={};self.requests={};self.last_request_ns=None

    def prefix_at(self,observed_ns):
        prefix=[]
        for slot in range(3):
            ns=observed_ns+slot*PERIOD
            candidates=[c for c in self.commitments if c.completed_ns<=observed_ns
                and c.plan.dispatch_ns<=ns<c.plan.expires_ns]
            if len(candidates)>1:raise ValueError('overlapping committed windows')
            command=(0.,0.,0.)
            if candidates:
                c=candidates[0]
                # Only a veto already known at the observation boundary can
                # change what this forecast treats as its committed prefix.
                veto=self.vetoes.get(c.plan.observed_ns)
                if veto is None or veto>observed_ns:command=c.plan.command
            prefix.append(tuple(command))
        return tuple(prefix)

    def commit(self,plan,completed_ns,prefix):
        prefix=tuple(tuple(c) for c in prefix)
        if (plan.dispatch_ns!=plan.observed_ns+3*PERIOD or plan.expires_ns!=plan.dispatch_ns+4*PERIOD
                or completed_ns>plan.dispatch_ns or prefix!=self.prefix_at(plan.observed_ns)):
            raise ValueError('on-time plan must preserve the known observation-time prefix')
        if any(max(c.plan.dispatch_ns,plan.dispatch_ns)<min(c.plan.expires_ns,plan.expires_ns)
                for c in self.commitments):raise ValueError('committed action windows overlap')
        self.commitments.append(Commitment(plan,completed_ns,prefix))

    def veto(self,plan,now_ns):
        self.vetoes.setdefault(plan.observed_ns,now_ns)

    def prefix_was_requested(self,plan):
        matches=[c for c in self.commitments if c.plan.observed_ns==plan.observed_ns]
        if len(matches)!=1:return False
        prefix=matches[0].prefix
        return all(self.requests.get(ns)==prefix[(ns-plan.observed_ns)//PERIOD]
            for ns in range(plan.observed_ns,plan.dispatch_ns,SERVICE))

    def record_request(self,now_ns,command):
        if self.last_request_ns is not None and now_ns-self.last_request_ns!=SERVICE:
            raise ValueError('complete ordered 20-ms request history required')
        self.last_request_ns=now_ns;self.requests[now_ns]=tuple(command)
        # Only three prefix intervals are checked; retain a larger bounded tail.
        self.requests={ns:c for ns,c in self.requests.items() if ns>=now_ns-2_000_000_000}
