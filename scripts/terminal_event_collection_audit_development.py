"""Full frozen raw audit followed by the prospective coverage/censoring layer."""
from lewm.terminal_event_coverage_development import classify_coverage
from scripts.independent_layout_collection_audit_development import audit_condition
from scripts.startup_raw_sensor_audit_development import read_json, read_npz


def audit_terminal_event_condition(directory, spec, result, definition):
    report, prefix, window, labels = audit_condition(directory, spec, result, definition)
    coverage = classify_coverage(spec, read_npz(directory, 'physics_trace.npz'),
        read_npz(directory, 'native_contacts.npz'), read_json(directory, 'contact_events.json'),
        read_json(directory, 'camera_audit.json'), read_json(directory, 'command_tape.json'),
        result, report, prefix, window, labels)
    return dict(report=report, prefix=prefix, coverage=coverage)
