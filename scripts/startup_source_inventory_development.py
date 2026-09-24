"""Narrow import discovery for new development sources; no export or tree scan."""
import ast
from pathlib import Path

from scripts.run_go2_successive_choice_maze_development_v1 import ROOT, digest


def allowed_relative(name):
    p = Path(name)
    if (p.is_absolute() or '..' in p.parts or not p.parts
            or any(s in ('sealed', 'sealed_test.json') or s.startswith('sealed_') for s in p.parts)):
        raise ValueError('nonprotected explicit repository-relative source required')
    return p


def local_sources(module):
    if not module or module.split('.')[0] not in ('lewm', 'scripts', 'lewm_genesis', 'lewm_worlds'): return []
    if any(not s.isidentifier() for s in module.split('.')): raise ValueError('valid local module name required')
    candidates = []
    relative = Path(*module.split('.'))
    for base in (Path('.'), Path('lewm_genesis'), Path('lewm_worlds')):
        for name in (base / relative.with_suffix('.py'), base / relative / '__init__.py'):
            name = allowed_relative(name); path = ROOT / name
            if path.is_file():
                if path.resolve() != path: raise ValueError('symlink source forbidden')
                candidates.append((module, str(name)))
    if len(candidates) > 1: raise ValueError('ambiguous local module source')
    return candidates


def discover_sources(seed_paths, inherited):
    """Stop at inherited bound sources; traverse only explicit new local imports.

The inherited manifest is not reclassified as a newly proved recursive closure.
All .ignore custody patterns are excluded before checking or reading paths.
"""
    discovered = {}; queue = []
    for name in seed_paths:
        path = allowed_relative(name)
        parts = list(path.with_suffix('').parts)
        if parts[:2] in (['lewm_genesis', 'lewm_genesis'], ['lewm_worlds', 'lewm_worlds']): parts.pop(0)
        if parts[-1] == '__init__': parts.pop()
        queue.append(('.'.join(parts), str(path)))
    while queue:
        module, name = queue.pop()
        if name in discovered or name in inherited: continue
        p = ROOT / allowed_relative(name)
        if p.resolve() != p or not p.is_file(): raise ValueError('existing nonsymlink source required: ' + name)
        discovered[name] = digest(p)
        if p.suffix != '.py': continue
        tree = ast.parse(p.read_text())
        for node in ast.walk(tree):
            modules = []
            if isinstance(node, ast.Import): modules = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom):
                if node.level:
                    package = module.split('.') if p.name == '__init__.py' else module.split('.')[:-1]
                    if node.level > len(package): raise ValueError('relative import outside local package')
                    prefix = package[:len(package) - node.level + 1]
                    base = '.'.join(prefix + ([node.module] if node.module else []))
                else: base = node.module or ''
                modules = [base] + [base + '.' + a.name for a in node.names if a.name != '*']
            for target in modules: queue.extend(local_sources(target))
    return inherited | discovered
