import hashlib
import importlib
import json
import pkgutil

import garak.probes
from garak.probes.base import Probe

from models import SeedCandidate, _PathEncoder


def extract_all_prompts() -> list[SeedCandidate]:
    """Walk the full garak probe library and collect every unique prompt.

    Iterates every importable module under garak.probes, instantiates each
    concrete Probe subclass, and harvests its .prompts list. Deduplicates by
    SHA-256 hash so no two SeedCandidates carry the same text.

    Returns
    -------
    list[SeedCandidate]
        One entry per unique prompt, annotated with its source probe, tags,
        goal string, and attack family (derived from the probe module name).
    """
    candidates: list[SeedCandidate] = []
    seen_hashes: set[str] = set()
    import_failures:  list[tuple[str, str]]        = []
    init_failures:    list[tuple[str, str, str]]   = []
    dynamic_probes:   list[tuple[str, str]]        = []

    for info in pkgutil.walk_packages(
        garak.probes.__path__, prefix=garak.probes.__name__ + "."
    ):
        modname = info.name
        try:
            module = importlib.import_module(modname)
        except Exception as e:
            import_failures.append((modname, str(e)))
            continue

        for attr_name in dir(module):
            cls = getattr(module, attr_name)
            if not (isinstance(cls, type)
                    and issubclass(cls, Probe)
                    and cls is not Probe):
                continue

            try:
                instance = cls()
            except Exception as e:
                init_failures.append((modname, attr_name, str(e)))
                continue

            prompts       = getattr(instance, "prompts", [])
            tags          = getattr(instance, "tags", [])
            goal          = getattr(instance, "goal", "")
            attack_family = modname.split(".")[-1]   # e.g. "garak.probes.dan" → "dan"

            if not prompts:
                dynamic_probes.append((modname, attr_name))
                continue

            for prompt in prompts:
                if isinstance(prompt, dict):
                    prompt_text = json.dumps(prompt, ensure_ascii=False, cls=_PathEncoder)
                elif isinstance(prompt, str):
                    prompt_text = prompt
                else:
                    continue   # skip unexpected types (bytes, None, …)

                h = hashlib.sha256(prompt_text.encode()).hexdigest()
                if h in seen_hashes:
                    continue
                seen_hashes.add(h)

                candidates.append(SeedCandidate(
                    prompt=prompt_text,
                    probe_module=modname,
                    probe_class=attr_name,
                    tags=tags,
                    goal=goal,
                    attack_family=attack_family,
                ))

    if import_failures:
        print(f"\n  [stage1] WARNING: {len(import_failures)} module(s) failed to import:")
        for mod, err in import_failures:
            print(f"    {mod}: {err}")

    if init_failures:
        print(f"\n  [stage1] WARNING: {len(init_failures)} probe(s) failed to instantiate:")
        for mod, cls_name, err in init_failures:
            print(f"    {mod}.{cls_name}: {err}")

    if dynamic_probes:
        print(f"\n  [stage1] WARNING: {len(dynamic_probes)} probe(s) had empty .prompts — "
              f"likely generate prompts dynamically via probe(generator) and were skipped:")
        for mod, cls_name in dynamic_probes:
            print(f"    {mod}.{cls_name}")

    print(f"\nExtracted {len(candidates)} unique prompts "
          f"from {len(set(c.attack_family for c in candidates))} attack families "
          f"({len(import_failures)} import failures, {len(init_failures)} init failures, "
          f"{len(dynamic_probes)} dynamic/empty skipped)")
    return candidates
