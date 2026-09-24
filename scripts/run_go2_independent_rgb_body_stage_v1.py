"""One fresh bounded collection followed by its terminal audit, including failure."""
import argparse
from scripts.independent_rgb_body_batch_development import BATCHES, output_root
from scripts.run_go2_independent_rgb_body_collection_v1 import run_batch
from scripts.audit_go2_independent_rgb_body_collection_v1 import run_audit


def run_stage(batch):
    output = output_root(batch)
    if output.exists() or output.is_symlink():
        raise ValueError('fresh stage only; never restart or resume an existing batch')
    collection_error = None
    try:
        run_batch(batch)
    except Exception as error:
        collection_error = error
    # Only this newly attempted stage may advance to its audit. A preflight
    # failure without a committed terminal record does not authorize an audit.
    terminals = [n for n in ('result.json', 'failure.json') if (output / n).is_file()]
    if len(terminals) == 1:
        run_audit(batch)
        print('RGB_BODY_STAGE_TERMINAL', batch, 'collection_failed', collection_error is not None,
              'terminal_audit_complete', True, flush=True)
    elif collection_error is None:
        raise ValueError('collection returned without exactly one terminal record')
    if collection_error is not None:
        raise collection_error


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--batch', choices=BATCHES, required=True)
    run_stage(parser.parse_args().batch)


if __name__ == '__main__': main()
