"""Pure provenance-order tests for the scorer-fit latent encoder."""
from __future__ import annotations

import hashlib
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from scripts import encode_go2_branch_corpus_v1_2 as encoder


class BranchCorpusEncoderProvenanceTests(unittest.TestCase):
    @staticmethod
    def _simple_receipt_tools():
        def validator(value):
            if not isinstance(value, dict) or set(value) != {"phase", "value"}:
                raise RuntimeError("synthetic receipt changed")
            return dict(value)

        def binding_builder(value, raw):
            return {
                "phase": value["phase"],
                "raw_sha256": hashlib.sha256(raw).hexdigest(),
                "byte_count": len(raw),
            }

        return validator, binding_builder

    @staticmethod
    def _transaction_fixture(directory, *, evidence_overrides=None):
        root = Path(directory) / "repo"
        out = root / encoder.V2_DESIGN.SCORER_FIT_RELATIVE_PATH
        target_relative = (
            encoder.V2_DESIGN.SCORER_FIT_RELATIVE_PATH
            / "latents_v2/horizon/fixture-candidate-0.f16")
        target = root / target_relative
        target.parent.mkdir(parents=True)
        with target.open("wb") as handle:
            handle.truncate(6_291_456)
        target.chmod(0o640)
        metadata = target.stat()
        target_binding = {
            "path": str(target_relative), "candidate_index": 0,
            "sha256": encoder.file_sha256(target),
            "byte_count": metadata.st_size, "shape": [4, 768, 1024],
            "device_id": metadata.st_dev, "inode": metadata.st_ino,
            "mode_octal": "0640", "link_count": 1,
        }
        immutable_contract = (
            encoder.V2_DESIGN.IMMUTABLE_SUCCESSOR_SCORER_CONTRACT_BINDING)
        lineage = {
            "scorer_fit_corpus_v2_scorer_contract_digest":
                immutable_contract["embedded_contract_self_digest"],
            "scorer_fit_corpus_v2_scorer_contract_artifact_digest":
                immutable_contract["self_digest"],
            "state_manifest_digest": "1" * 64,
            "full_bank_assignment_manifest_digest": "2" * 64,
            "corpus_digest": "3" * 64,
            "branch_smoke_receipt_digest": "4" * 64,
            "encoder_compute_dtype_correction_digest":
                encoder.V2_DESIGN.
                IMMUTABLE_ENCODER_COMPUTE_DTYPE_CORRECTION_DIGEST,
            "encoder_path_projection_correction_digest": "5" * 64,
        }
        evidence = {
            "latent_index_digest": "6" * 64,
            "encoding_smoke_receipt_digest": "7" * 64,
            "registered_smoke_shard_inventory_digest": "8" * 64,
            "registered_smoke_non_target_shard_inventory_digest": "9" * 64,
            "registered_smoke_non_target_shard_custody_inventory_digest":
                "d" * 64,
            "registered_smoke_stable_artifact_inventory_digest": "a" * 64,
            "zero_new_resume_verified": True,
            **(evidence_overrides or {}),
        }
        prepared = (
            encoder.V2_DESIGN.
            build_full_bank_v2_smoke_regeneration_prepared_receipt(
                lineage=lineage, designated_target=target_binding,
                pretransaction_evidence=evidence))
        contract = (
            encoder.V2_DESIGN.
            ENCODER_PATH_PROJECTION_SINGLE_SHARD_REGENERATION_TRANSACTION_CONTRACT)
        return {
            "root": root, "out": out, "target": target,
            "target_binding": target_binding, "lineage": lineage,
            "evidence": evidence, "prepared": prepared,
            "contract": contract,
            "contract_digest": encoder.V2_DESIGN.canonical_digest(contract),
            "path_digest": lineage[
                "encoder_path_projection_correction_digest"],
        }

    @staticmethod
    def _publish_transaction_receipt(
            fixture, phase, payload, *, link_side_effect=None):
        paths = encoder._transaction_paths(  # noqa: SLF001
            fixture["out"], fixture["contract"])
        if phase == "PREPARED":
            validator = (
                encoder.V2_DESIGN.
                validate_full_bank_v2_smoke_regeneration_prepared_receipt)
            builder = (
                encoder.V2_DESIGN.
                full_bank_v2_smoke_regeneration_prepared_receipt_artifact_binding)
            final = paths["prepared"]
            staged = paths["prepared_staged"]
        else:
            validator = (
                encoder.V2_DESIGN.
                validate_full_bank_v2_smoke_regeneration_complete_receipt)
            builder = (
                encoder.V2_DESIGN.
                full_bank_v2_smoke_regeneration_complete_receipt_artifact_binding)
            final = paths["complete"]
            staged = paths["complete_staged"]
        context = (mock.patch.object(
            encoder.os, "link", side_effect=link_side_effect)
            if link_side_effect is not None else mock.patch.object(
                encoder.os, "link", wraps=os.link))
        with context:
            return encoder._publish_immutable_json_no_overwrite(  # noqa: SLF001
                final_path=final, staged_path=staged, payload=payload,
                validator=validator, binding_builder=builder,
                label=phase, recover_nonexact_staged=True)

    @staticmethod
    def _transaction_status(fixture):
        return encoder._classify_full_bank_v2_single_shard_regeneration_transaction(  # noqa: E501, SLF001
            out=fixture["out"],
            encoder_path_projection_correction_digest=
                fixture["path_digest"],
            transaction_contract=fixture["contract"],
            transaction_contract_digest=fixture["contract_digest"])

    @staticmethod
    def _restore_transaction_target(fixture):
        target = fixture["target"]
        with target.open("wb") as handle:
            handle.truncate(fixture["target_binding"]["byte_count"])
        target.chmod(int(fixture["target_binding"]["mode_octal"], 8))

    @staticmethod
    def _build_transaction_complete(fixture, smoke, *, new_horizons=1):
        prepared_raw = encoder._pretty_json_bytes(  # noqa: SLF001
            fixture["prepared"])
        prepared_binding = (
            encoder.V2_DESIGN.
            full_bank_v2_smoke_regeneration_prepared_receipt_artifact_binding(
                fixture["prepared"], prepared_raw))
        regenerated = encoder._target_stat_binding(  # noqa: SLF001
            fixture["target"],
            logical_path=fixture["target_binding"]["path"],
            candidate_index=0,
            sha256=fixture["target_binding"]["sha256"],
            byte_count=fixture["target_binding"]["byte_count"],
            shape=fixture["target_binding"]["shape"])
        return (
            encoder.V2_DESIGN.
            build_full_bank_v2_smoke_regeneration_complete_receipt(
                prepared_receipt_binding=prepared_binding,
                lineage=fixture["lineage"],
                designated_target=fixture["target_binding"],
                retained_backup_binding=fixture["prepared"][
                    "expected_backup_binding"],
                regenerated_target_binding=regenerated,
                non_target_shard_inventory_digest=fixture["evidence"][
                    "registered_smoke_non_target_shard_inventory_digest"],
                posttransaction_evidence={
                    "latent_index_digest": fixture["evidence"][
                        "latent_index_digest"],
                    "encoding_smoke_receipt_digest": smoke[
                        "smoke_receipt_digest"],
                    "registered_smoke_shard_inventory_digest":
                        fixture["evidence"][
                            "registered_smoke_shard_inventory_digest"],
                    "registered_smoke_non_target_shard_custody_inventory_digest":
                        fixture["evidence"][
                            "registered_smoke_non_target_shard_custody_inventory_digest"],
                    "registered_smoke_stable_artifact_inventory_digest":
                        fixture["evidence"][
                            "registered_smoke_stable_artifact_inventory_digest"],
                    "encoder_invocation_new_context_shards": 0,
                    "encoder_invocation_new_horizon_shards": new_horizons,
                    "target_restored_exact": True,
                    "non_target_shards_unchanged": True,
                    "complete_before_pass_smoke": True,
                },
                final_smoke_receipt_binding=
                    encoder._smoke_receipt_artifact_binding(smoke)))  # noqa: SLF001

    @staticmethod
    def _protocol_smoke(fixture, prepared_binding, *, new_horizons=1):
        smoke = {
            "schema": encoder.FULL_BANK_V2_SMOKE_SCHEMA,
            "status": encoder.STATUS,
            "pass": True,
            "zero_new_resume_verified": True,
            "single_shard_deletion_regeneration_verified": True,
            "smoke_protocol_complete": True,
            "single_shard_regeneration_transaction_contract_digest":
                fixture["contract_digest"],
            "single_shard_regeneration_prepared_digest":
                prepared_binding["self_digest"],
            "single_shard_regeneration_transaction_complete": True,
            "single_shard_regeneration_target_atomic_move_count": 1,
            "single_shard_regeneration_target_regeneration_count": 1,
            "latent_index_digest": fixture["evidence"][
                "latent_index_digest"],
            "invocation_new_context_shards": 0,
            "invocation_new_horizon_shards": new_horizons,
            **fixture["lineage"],
        }
        smoke["smoke_receipt_digest"] = encoder.canonical_digest(smoke)
        return smoke

    def test_immutable_receipt_link_publication_order_and_recovery(self):
        validator, binding_builder = self._simple_receipt_tools()
        for phase in ("PREPARED", "COMPLETE"):
            with self.subTest(phase=phase), tempfile.TemporaryDirectory() as directory:
                parent = Path(directory) / phase.lower()
                parent.mkdir()
                final = parent / f"{phase.lower()}.json"
                staged = parent / f"{phase.lower()}.json.staged"
                payload = {"phase": phase, "value": 1}
                events = []
                real_link = os.link
                real_unlink = os.unlink
                real_fsync_directory = encoder._fsync_directory  # noqa: SLF001

                def ordered_link(source, destination, **kwargs):
                    events.append("link")
                    return real_link(source, destination, **kwargs)

                def ordered_unlink(path, *args, **kwargs):
                    if Path(path) == staged:
                        events.append("unlink-stage")
                    return real_unlink(path, *args, **kwargs)

                def ordered_directory_fsync(path):
                    events.append("fsync-directory")
                    return real_fsync_directory(path)

                with mock.patch.object(
                        encoder.os, "link", side_effect=ordered_link), \
                        mock.patch.object(
                            encoder.os, "unlink", side_effect=ordered_unlink), \
                        mock.patch.object(
                            encoder, "_fsync_directory",  # noqa: SLF001
                            side_effect=ordered_directory_fsync):
                    encoder._publish_immutable_json_no_overwrite(  # noqa: SLF001
                        final_path=final, staged_path=staged,
                        payload=payload, validator=validator,
                        binding_builder=binding_builder, label=phase,
                        recover_nonexact_staged=False)
                self.assertEqual(events, [
                    "link", "fsync-directory", "unlink-stage",
                    "fsync-directory",
                ])
                self.assertTrue(final.is_file())
                self.assertFalse(staged.exists())

                final.unlink()
                interrupted = []

                def link_then_interrupt(source, destination, **kwargs):
                    real_link(source, destination, **kwargs)
                    interrupted.append("after-link")
                    raise OSError("synthetic post-link interruption")

                with mock.patch.object(
                        encoder.os, "link", side_effect=link_then_interrupt):
                    with self.assertRaisesRegex(
                            OSError, "post-link interruption"):
                        encoder._publish_immutable_json_no_overwrite(  # noqa: SLF001
                            final_path=final, staged_path=staged,
                            payload=payload, validator=validator,
                            binding_builder=binding_builder, label=phase,
                            recover_nonexact_staged=False)
                self.assertEqual(interrupted, ["after-link"])
                final_stat = final.stat()
                staged_stat = staged.stat()
                self.assertEqual(
                    (final_stat.st_dev, final_stat.st_ino),
                    (staged_stat.st_dev, staged_stat.st_ino))
                recovery_fsyncs = []
                with mock.patch.object(
                        encoder, "_fsync_directory",  # noqa: SLF001
                        side_effect=lambda path: (
                            recovery_fsyncs.append(Path(path)),
                            real_fsync_directory(path))[1]):
                    encoder._publish_immutable_json_no_overwrite(  # noqa: SLF001
                        final_path=final, staged_path=staged,
                        payload=payload, validator=validator,
                        binding_builder=binding_builder, label=phase,
                        recover_nonexact_staged=False)
                self.assertFalse(staged.exists())
                self.assertEqual(len(recovery_fsyncs), 2)
                self.assertEqual(final.stat().st_ino, final_stat.st_ino)

                # A staged-only exact receipt from a pre-link crash must be
                # reopened and fsynced before its no-overwrite link.
                final.unlink()
                staged.write_bytes(encoder._pretty_json_bytes(payload))  # noqa: SLF001
                staged.chmod(0o444)
                staged_inode = staged.stat().st_ino
                ordered = []
                real_fsync = os.fsync

                def observed_fsync(descriptor):
                    ordered.append(("fsync", os.fstat(descriptor).st_ino))
                    return real_fsync(descriptor)

                def observed_link(source, destination, **kwargs):
                    ordered.append(("link", Path(source).stat().st_ino))
                    return real_link(source, destination, **kwargs)

                with mock.patch.object(
                        encoder.os, "fsync", side_effect=observed_fsync), \
                        mock.patch.object(
                            encoder.os, "link", side_effect=observed_link):
                    encoder._publish_immutable_json_no_overwrite(  # noqa: SLF001
                        final_path=final, staged_path=staged,
                        payload=payload, validator=validator,
                        binding_builder=binding_builder, label=phase,
                        recover_nonexact_staged=False)
                staged_fsync = ordered.index(("fsync", staged_inode))
                staged_link = ordered.index(("link", staged_inode))
                self.assertLess(staged_fsync, staged_link)

    def test_immutable_receipt_partial_stage_recovery_is_narrow_and_durable(self):
        validator, binding_builder = self._simple_receipt_tools()
        for phase in ("PREPARED", "COMPLETE"):
            with self.subTest(phase=phase), tempfile.TemporaryDirectory() as directory:
                parent = Path(directory)
                final = parent / f"{phase.lower()}.json"
                staged = parent / f"{phase.lower()}.json.staged"
                staged.write_bytes(b"partial")
                staged.chmod(0o444)
                payload = {"phase": phase, "value": 2}
                with self.assertRaisesRegex(RuntimeError, "partial or nonexact"):
                    encoder._publish_immutable_json_no_overwrite(  # noqa: SLF001
                        final_path=final, staged_path=staged,
                        payload=payload, validator=validator,
                        binding_builder=binding_builder, label=phase,
                        recover_nonexact_staged=False)
                self.assertEqual(staged.read_bytes(), b"partial")

                events = []
                real_unlink = os.unlink
                real_fsync_directory = encoder._fsync_directory  # noqa: SLF001

                def ordered_unlink(path, *args, **kwargs):
                    if Path(path) == staged:
                        events.append("unlink-stage")
                    return real_unlink(path, *args, **kwargs)

                def ordered_directory_fsync(path):
                    events.append("fsync-directory")
                    return real_fsync_directory(path)

                with mock.patch.object(
                        encoder.os, "unlink", side_effect=ordered_unlink), \
                        mock.patch.object(
                            encoder, "_fsync_directory",  # noqa: SLF001
                            side_effect=ordered_directory_fsync):
                    encoder._publish_immutable_json_no_overwrite(  # noqa: SLF001
                        final_path=final, staged_path=staged,
                        payload=payload, validator=validator,
                        binding_builder=binding_builder, label=phase,
                        recover_nonexact_staged=True)
                self.assertEqual(events[:2], [
                    "unlink-stage", "fsync-directory"])
                self.assertTrue(final.is_file())
                self.assertFalse(staged.exists())

    def test_no_gap_successor_publication_recovers_every_material_boundary(self):
        old = {"generation": "old"}
        new = {"generation": "new"}
        old_raw = encoder._pretty_json_bytes(old)  # noqa: SLF001
        new_raw = encoder._pretty_json_bytes(new)  # noqa: SLF001
        for boundary in ("archive-link", "before-replace", "after-replace"):
            with self.subTest(boundary=boundary), \
                    tempfile.TemporaryDirectory() as directory:
                parent = Path(directory)
                active = parent / "smoke.json"
                archive_dir = parent / "archive"
                active.write_bytes(old_raw)
                real_link = os.link
                real_replace = os.replace

                if boundary == "archive-link":
                    def interrupted_link(source, destination, **kwargs):
                        real_link(source, destination, **kwargs)
                        raise OSError("synthetic archive-link interruption")

                    patcher = mock.patch.object(
                        encoder.os, "link", side_effect=interrupted_link)
                elif boundary == "before-replace":
                    patcher = mock.patch.object(
                        encoder.os, "replace",
                        side_effect=OSError(
                            "synthetic pre-replace interruption"))
                else:
                    def replace_then_interrupt(source, destination):
                        real_replace(source, destination)
                        raise OSError("synthetic post-replace interruption")

                    patcher = mock.patch.object(
                        encoder.os, "replace",
                        side_effect=replace_then_interrupt)
                with mock.patch.object(encoder, "ROOT", parent), patcher, \
                        self.assertRaisesRegex(OSError, "synthetic"):
                    encoder._publish_json_with_archive_no_gap(  # noqa: SLF001
                        active_path=active, payload=new,
                        archive_dir=archive_dir, label="smoke")
                self.assertTrue(active.is_file())
                self.assertIn(active.read_bytes(), (old_raw, new_raw))

                staged = active.with_name(".smoke.json.successor-staged")
                staged_inode = staged.stat().st_ino if staged.exists() else None
                active_inode_before_resume = active.stat().st_ino
                parent_inode = active.parent.stat().st_ino
                events = []
                real_fsync = os.fsync

                def observed_fsync(descriptor):
                    try:
                        inode = os.fstat(descriptor).st_ino
                    except OSError:
                        inode = None
                    events.append(("fsync", inode))
                    return real_fsync(descriptor)

                def observed_replace(source, destination):
                    events.append(("replace", Path(source).stat().st_ino))
                    return real_replace(source, destination)

                with mock.patch.object(
                        encoder.os, "fsync", side_effect=observed_fsync), \
                        mock.patch.object(
                            encoder.os, "replace", side_effect=observed_replace), \
                        mock.patch.object(encoder, "ROOT", parent):
                    encoder._publish_json_with_archive_no_gap(  # noqa: SLF001
                        active_path=active, payload=new,
                        archive_dir=archive_dir, label="smoke")
                self.assertEqual(active.read_bytes(), new_raw)
                self.assertFalse(staged.exists())
                if staged_inode is not None:
                    stage_fsync_positions = [
                        index for index, event in enumerate(events)
                        if event == ("fsync", staged_inode)]
                    replace_positions = [
                        index for index, event in enumerate(events)
                        if event[0] == "replace"]
                    self.assertTrue(stage_fsync_positions)
                    self.assertTrue(replace_positions)
                    self.assertLess(
                        stage_fsync_positions[-1], replace_positions[0])
                if boundary == "after-replace":
                    self.assertIn(
                        ("fsync", active_inode_before_resume), events)
                    self.assertIn(("fsync", parent_inode), events)

    def test_target_move_is_no_replace_and_durably_ordered(self):
        def fixture(directory):
            root = Path(directory)
            source_dir = root / "active"
            destination_dir = root / "transaction"
            source_dir.mkdir()
            destination_dir.mkdir()
            target = source_dir / "candidate-0.f16"
            backup = destination_dir / "backup.f16"
            target.write_bytes(b"synthetic-immutable-target")
            target.chmod(0o640)
            metadata = target.stat()
            binding = {
                "path": "fixture", "candidate_index": 0,
                "sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
                "byte_count": metadata.st_size, "shape": [1],
                "device_id": metadata.st_dev, "inode": metadata.st_ino,
                "mode_octal": "0640", "link_count": 1,
            }
            return target, backup, binding

        with tempfile.TemporaryDirectory() as directory:
            target, backup, binding = fixture(directory)
            events = []
            real_rename = encoder._rename_noreplace  # noqa: SLF001
            real_file_fsync = encoder._fsync_exact_bound_regular_file  # noqa: SLF001
            real_dir_fsync = encoder._fsync_directory  # noqa: SLF001

            def ordered_rename(source, destination):
                events.append("rename-noreplace")
                return real_rename(source, destination)

            def ordered_file_fsync(path, expected, *, label):
                events.append("backup-reopen-fsync")
                return real_file_fsync(path, expected, label=label)

            def ordered_dir_fsync(path):
                events.append(
                    "destination-dir-fsync" if Path(path) == backup.parent
                    else "source-dir-fsync")
                return real_dir_fsync(path)

            with mock.patch.object(
                    encoder, "_rename_noreplace",  # noqa: SLF001
                    side_effect=ordered_rename), \
                    mock.patch.object(
                        encoder, "_fsync_exact_bound_regular_file",  # noqa: SLF001
                        side_effect=ordered_file_fsync), \
                    mock.patch.object(
                        encoder, "_fsync_directory",  # noqa: SLF001
                        side_effect=ordered_dir_fsync):
                encoder._atomic_move_target_to_backup_no_replace(  # noqa: SLF001
                    target_path=target, backup_path=backup,
                    target_binding=binding)
            self.assertEqual(events, [
                "rename-noreplace", "backup-reopen-fsync",
                "destination-dir-fsync", "source-dir-fsync",
            ])
            self.assertFalse(target.exists())
            self.assertEqual(backup.stat().st_ino, binding["inode"])

        with tempfile.TemporaryDirectory() as directory:
            target, backup, binding = fixture(directory)

            def destination_race(_source, destination):
                Path(destination).write_bytes(b"racing-destination")
                raise FileExistsError("synthetic destination race")

            with mock.patch.object(
                    encoder, "_rename_noreplace",  # noqa: SLF001
                    side_effect=destination_race), \
                    self.assertRaisesRegex(FileExistsError, "destination race"):
                encoder._atomic_move_target_to_backup_no_replace(  # noqa: SLF001
                    target_path=target, backup_path=backup,
                    target_binding=binding)
            self.assertEqual(target.read_bytes(), b"synthetic-immutable-target")
            self.assertEqual(backup.read_bytes(), b"racing-destination")

    def test_target_move_fault_boundaries_leave_one_exact_recovery_copy(self):
        for boundary in ("after-rename", "after-destination-fsync",
                         "after-source-fsync"):
            with self.subTest(boundary=boundary), \
                    tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                target = root / "active/candidate-0.f16"
                backup = root / "transaction/backup.f16"
                target.parent.mkdir()
                backup.parent.mkdir()
                raw = b"synthetic-immutable-target"
                target.write_bytes(raw)
                target.chmod(0o640)
                metadata = target.stat()
                binding = {
                    "path": "fixture", "candidate_index": 0,
                    "sha256": hashlib.sha256(raw).hexdigest(),
                    "byte_count": len(raw), "shape": [1],
                    "device_id": metadata.st_dev, "inode": metadata.st_ino,
                    "mode_octal": "0640", "link_count": 1,
                }
                real_rename = encoder._rename_noreplace  # noqa: SLF001
                real_dir_fsync = encoder._fsync_directory  # noqa: SLF001

                def interrupted_rename(source, destination):
                    real_rename(source, destination)
                    if boundary == "after-rename":
                        raise OSError("synthetic move interruption")

                directory_fsync_count = 0

                def interrupted_dir_fsync(path):
                    nonlocal directory_fsync_count
                    real_dir_fsync(path)
                    directory_fsync_count += 1
                    if ((boundary == "after-destination-fsync"
                         and directory_fsync_count == 1)
                            or (boundary == "after-source-fsync"
                                and directory_fsync_count == 2)):
                        raise OSError("synthetic move interruption")

                with mock.patch.object(
                        encoder, "_rename_noreplace",  # noqa: SLF001
                        side_effect=interrupted_rename), \
                        mock.patch.object(
                            encoder, "_fsync_directory",  # noqa: SLF001
                            side_effect=interrupted_dir_fsync), \
                        self.assertRaisesRegex(
                            OSError, "synthetic move interruption"):
                    encoder._atomic_move_target_to_backup_no_replace(  # noqa: SLF001
                        target_path=target, backup_path=backup,
                        target_binding=binding)
                self.assertFalse(target.exists())
                self.assertEqual(backup.read_bytes(), raw)
                self.assertEqual(backup.stat().st_ino, binding["inode"])

    def test_non_target_custody_digest_preserves_atime_and_detects_stat_change(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "repo"
            out = root / encoder.V2_DESIGN.SCORER_FIT_RELATIVE_PATH
            records = []
            for candidate_index in range(13):
                relative = Path("latents_v2/horizon") / (
                    f"candidate-{candidate_index}.f16")
                path = out / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                raw = f"synthetic-{candidate_index}".encode()
                path.write_bytes(raw)
                path.chmod(0o640)
                old_atime = 1_700_000_000_000_000_000 + candidate_index
                os.utime(path, ns=(old_atime, path.stat().st_mtime_ns))
                records.append({
                    "path": str(
                        encoder.V2_DESIGN.SCORER_FIT_RELATIVE_PATH / relative),
                    "sha256": hashlib.sha256(raw).hexdigest(),
                    "byte_count": len(raw), "shape": [4, 768, 1024],
                })
            target_path = records[0]["path"]
            before = {
                item["path"]: (root / item["path"]).stat().st_atime_ns
                for item in records
            }
            with mock.patch.object(encoder, "ROOT", root):
                first = (
                    encoder._non_target_smoke_shard_custody_inventory_digest(  # noqa: E501, SLF001
                        records, out=out, target_path=target_path))
                after = {
                    item["path"]: (root / item["path"]).stat().st_atime_ns
                    for item in records
                }
                self.assertEqual(after, before)
                changed = root / records[1]["path"]
                changed.chmod(0o600)
                second = (
                    encoder._non_target_smoke_shard_custody_inventory_digest(  # noqa: E501, SLF001
                        records, out=out, target_path=target_path))
                self.assertNotEqual(second, first)

                regular = root / records[2]["path"]
                symlink = regular.with_name("symlink.f16")
                symlink.symlink_to(regular)
                with self.assertRaisesRegex(
                        RuntimeError, "without changing atime"):
                    encoder._file_sha256_without_atime_change(symlink)  # noqa: SLF001

    def test_transaction_classifier_covers_all_recovery_states_and_custody(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture = self._transaction_fixture(directory)
            with mock.patch.object(encoder, "ROOT", fixture["root"]), \
                    mock.patch.object(
                        encoder, "_transaction_lineage",  # noqa: SLF001
                        return_value=fixture["lineage"]):
                status = self._transaction_status(fixture)
                expected_keys = {
                    "transaction_state", "prepared_present",
                    "prepared_receipt_digest", "target_state",
                    "backup_state", "complete_present",
                    "complete_receipt_digest", "pass_smoke_state",
                    "next_action",
                    "encoder_path_projection_correction_digest",
                    "single_shard_regeneration_transaction_contract_digest",
                    "prepared_staged_state", "complete_staged_state",
                    "target_exact", "backup_exact",
                    "target_backup_custody_exact",
                    "regenerated_target_custody_exact",
                    "candidate_outcomes_used_for_selection",
                    "final_200_state_corpus_generated",
                }
                self.assertEqual(set(status), expected_keys)
                self.assertEqual(status["transaction_state"], "UNSTARTED")
                self.assertEqual(status["target_state"], "NOT_APPLICABLE")

                paths = encoder._transaction_paths(  # noqa: SLF001
                    fixture["out"], fixture["contract"])
                paths["prepared_staged"].parent.mkdir(parents=True)
                paths["prepared_staged"].write_bytes(
                    encoder._pretty_json_bytes(fixture["prepared"]))  # noqa: SLF001
                paths["prepared_staged"].chmod(0o444)
                status = self._transaction_status(fixture)
                self.assertEqual(status["transaction_state"], "UNSTARTED")
                self.assertEqual(status["prepared_staged_state"], "EXACT")

                prepared_binding = self._publish_transaction_receipt(
                    fixture, "PREPARED", fixture["prepared"])
                status = self._transaction_status(fixture)
                self.assertEqual(
                    (status["transaction_state"], status["target_state"],
                     status["backup_state"], status["next_action"]),
                    ("PREPARED_MOVE_PENDING", "EXACT", "ABSENT",
                     "ATOMIC_MOVE_ONCE"))

                encoder._atomic_move_target_to_backup_no_replace(  # noqa: SLF001
                    target_path=fixture["target"],
                    backup_path=paths["backup"],
                    target_binding=fixture["target_binding"])
                status = self._transaction_status(fixture)
                self.assertEqual(
                    (status["transaction_state"], status["target_state"],
                     status["backup_state"], status["next_action"]),
                    ("MOVED_REGENERATION_PENDING", "ABSENT", "EXACT",
                     "RUN_REGENERATION_ENCODER_ONCE"))

                self._restore_transaction_target(fixture)
                status = self._transaction_status(fixture)
                self.assertEqual(
                    (status["transaction_state"], status["target_state"],
                     status["backup_state"], status["next_action"]),
                    ("RESTORED_COMPLETE_PENDING", "EXACT", "EXACT",
                     "CREATE_COMPLETE_WITHOUT_SECOND_MOVE_OR_REGENERATION"))

                smoke = self._protocol_smoke(fixture, prepared_binding)
                complete = self._build_transaction_complete(fixture, smoke)
                paths["complete_staged"].write_bytes(
                    encoder._pretty_json_bytes(complete))  # noqa: SLF001
                paths["complete_staged"].chmod(0o444)
                status = self._transaction_status(fixture)
                self.assertEqual(
                    status["transaction_state"], "RESTORED_COMPLETE_PENDING")
                self.assertEqual(status["complete_staged_state"], "EXACT")

                complete_binding = self._publish_transaction_receipt(
                    fixture, "COMPLETE", complete)
                status = self._transaction_status(fixture)
                self.assertEqual(
                    status["transaction_state"],
                    "COMPLETE_SMOKE_PUBLICATION_PENDING")
                smoke_path = fixture["out"] / encoder.FULL_BANK_V2_SMOKE_NAME
                smoke_path.write_bytes(encoder._pretty_json_bytes(smoke))  # noqa: SLF001
                status = self._transaction_status(fixture)
                self.assertEqual(status["transaction_state"], "COMPLETE")
                self.assertEqual(
                    status["pass_smoke_state"],
                    "EXACT_BOUND_PROTOCOL_PASS")
                self.assertEqual(
                    status["next_action"], "NO_TRANSACTION_MUTATION")
                self.assertEqual(
                    status["complete_receipt_digest"],
                    complete_binding["self_digest"])

                # Exact semantic bytes are insufficient after COMPLETE: the
                # regenerated inode recorded by COMPLETE is part of custody.
                third = fixture["target"].with_suffix(".third")
                with third.open("wb") as handle:
                    handle.truncate(fixture["target_binding"]["byte_count"])
                third.chmod(0o640)
                os.replace(third, fixture["target"])
                with self.assertRaisesRegex(
                        RuntimeError, "COMPLETE custody changed"):
                    self._transaction_status(fixture)

    def test_transaction_classifier_requires_final_stage_hardlink_custody(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture = self._transaction_fixture(directory)
            with mock.patch.object(encoder, "ROOT", fixture["root"]):
                real_link = os.link

                def link_then_interrupt(source, destination, **kwargs):
                    real_link(source, destination, **kwargs)
                    raise OSError("synthetic receipt-link interruption")

                with self.assertRaisesRegex(OSError, "receipt-link"):
                    self._publish_transaction_receipt(
                        fixture, "PREPARED", fixture["prepared"],
                        link_side_effect=link_then_interrupt)
                paths = encoder._transaction_paths(  # noqa: SLF001
                    fixture["out"], fixture["contract"])
                status = self._transaction_status(fixture)
                self.assertEqual(
                    status["transaction_state"], "PREPARED_MOVE_PENDING")
                self.assertEqual(status["prepared_staged_state"], "EXACT")
                self.assertEqual(
                    paths["prepared"].stat().st_ino,
                    paths["prepared_staged"].stat().st_ino)

                raw = paths["prepared_staged"].read_bytes()
                paths["prepared_staged"].unlink()
                paths["prepared_staged"].write_bytes(raw)
                paths["prepared_staged"].chmod(0o444)
                with self.assertRaisesRegex(
                        RuntimeError, "final/staged custody changed"):
                    self._transaction_status(fixture)

    def test_complete_classifier_accepts_refreshed_pass_only_with_exact_archive(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture = self._transaction_fixture(directory)
            with mock.patch.object(encoder, "ROOT", fixture["root"]):
                prepared_binding = self._publish_transaction_receipt(
                    fixture, "PREPARED", fixture["prepared"])
                paths = encoder._transaction_paths(  # noqa: SLF001
                    fixture["out"], fixture["contract"])
                encoder._atomic_move_target_to_backup_no_replace(  # noqa: SLF001
                    target_path=fixture["target"],
                    backup_path=paths["backup"],
                    target_binding=fixture["target_binding"])
                self._restore_transaction_target(fixture)
                protocol_smoke = self._protocol_smoke(
                    fixture, prepared_binding)
                complete = self._build_transaction_complete(
                    fixture, protocol_smoke)
                complete_binding = self._publish_transaction_receipt(
                    fixture, "COMPLETE", complete)
                smoke_path = fixture["out"] / encoder.FULL_BANK_V2_SMOKE_NAME
                smoke_path.write_bytes(
                    encoder._pretty_json_bytes(protocol_smoke))  # noqa: SLF001
                current_corpus_digest = "e" * 64
                current_branch_smoke_digest = "f" * 64
                index = {
                    "complete": True,
                    "scorer_fit_corpus_v2_scorer_contract_digest":
                        fixture["lineage"][
                            "scorer_fit_corpus_v2_scorer_contract_digest"],
                    "scorer_fit_corpus_v2_scorer_contract_artifact_digest":
                        fixture["lineage"][
                            "scorer_fit_corpus_v2_scorer_contract_artifact_digest"],
                    "state_manifest_digest": fixture["lineage"][
                        "state_manifest_digest"],
                    "full_bank_assignment_manifest_digest":
                        fixture["lineage"][
                            "full_bank_assignment_manifest_digest"],
                    "corpus_digest": current_corpus_digest,
                    "encoder_compute_dtype_correction_digest":
                        fixture["lineage"][
                            "encoder_compute_dtype_correction_digest"],
                    "encoder_path_projection_correction_digest":
                        fixture["path_digest"],
                }
                index["latents_index_digest"] = encoder.canonical_digest(index)
                (fixture["out"] / encoder.FULL_BANK_V2_INDEX_NAME).write_bytes(
                    encoder._pretty_json_bytes(index))  # noqa: SLF001
                refreshed = {
                    "schema": encoder.FULL_BANK_V2_SMOKE_SCHEMA,
                    "status": encoder.STATUS, "pass": True,
                    "index_scope": "complete_full_corpus",
                    "latent_index_digest": index["latents_index_digest"],
                    "corpus_digest": current_corpus_digest,
                    "branch_smoke_receipt_digest":
                        current_branch_smoke_digest,
                    "single_shard_regeneration_prepared_digest":
                        prepared_binding["self_digest"],
                    "single_shard_regeneration_complete_digest":
                        complete_binding["self_digest"],
                    **{
                        key: fixture["lineage"][key] for key in (
                            "scorer_fit_corpus_v2_scorer_contract_digest",
                            "scorer_fit_corpus_v2_scorer_contract_artifact_digest",
                            "state_manifest_digest",
                            "full_bank_assignment_manifest_digest",
                            "encoder_compute_dtype_correction_digest",
                            "encoder_path_projection_correction_digest",
                        )
                    },
                }
                refreshed["smoke_receipt_digest"] = (
                    encoder.canonical_digest(refreshed))
                encoder._publish_json_with_archive_no_gap(  # noqa: SLF001
                    active_path=smoke_path, payload=refreshed,
                    archive_dir=fixture["out"] /
                        encoder.FULL_BANK_V2_SUPERSEDED_RECEIPTS_NAME,
                    label="smoke")
                current_bundle = {
                    "receipt": {"corpus_digest": current_corpus_digest},
                    "branch_smoke": {
                        "smoke_branch_receipt_digest":
                            current_branch_smoke_digest},
                }
                with mock.patch.object(
                        encoder.CORPUS_BUILDER,
                        "load_and_validate_full_bank_v2_branch_outputs_for_consumption",
                        return_value=current_bundle):
                    status = self._transaction_status(fixture)
                self.assertEqual(status["transaction_state"], "COMPLETE")
                self.assertEqual(
                    status["pass_smoke_state"],
                    "VALID_REFRESHED_PASS_WITH_EXACT_PROTOCOL_PASS_ARCHIVE")
                tampered = {
                    **{key: value for key, value in refreshed.items()
                       if key != "smoke_receipt_digest"},
                    "corpus_digest": "0" * 64,
                }
                tampered["smoke_receipt_digest"] = (
                    encoder.canonical_digest(tampered))
                smoke_path.write_bytes(
                    encoder._pretty_json_bytes(tampered))  # noqa: SLF001
                with mock.patch.object(
                        encoder.CORPUS_BUILDER,
                        "load_and_validate_full_bank_v2_branch_outputs_for_consumption",
                        return_value=current_bundle), \
                        self.assertRaisesRegex(
                            RuntimeError, "smoke/index lineage"):
                    self._transaction_status(fixture)
                smoke_path.write_bytes(
                    encoder._pretty_json_bytes(refreshed))  # noqa: SLF001
                smoke_path.unlink()
                status = self._transaction_status(fixture)
                self.assertEqual(
                    status["transaction_state"],
                    "COMPLETE_SMOKE_PUBLICATION_PENDING")

    def test_precomplete_live_lineage_and_staged_phase_fail_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture = self._transaction_fixture(directory)
            with mock.patch.object(encoder, "ROOT", fixture["root"]):
                paths = encoder._transaction_paths(  # noqa: SLF001
                    fixture["out"], fixture["contract"])
                paths["complete_staged"].parent.mkdir(parents=True)
                paths["complete_staged"].write_bytes(b"partial-complete")
                with self.assertRaisesRegex(
                        RuntimeError, "without PREPARED"):
                    self._transaction_status(fixture)
                paths["complete_staged"].unlink()

                self._publish_transaction_receipt(
                    fixture, "PREPARED", fixture["prepared"])
                wrong_lineage = {
                    **fixture["lineage"], "corpus_digest": "f" * 64}
                with self.assertRaisesRegex(
                        RuntimeError, "differs from the live corpus"):
                    encoder._validate_transaction_live_lineage(  # noqa: SLF001
                        out=fixture["out"],
                        transaction_contract=fixture["contract"],
                        expected_lineage=wrong_lineage)

                paths["complete_staged"].write_bytes(b"partial-complete")
                with self.assertRaisesRegex(
                        RuntimeError, "outside its safe phase"):
                    self._transaction_status(fixture)

    def test_prepared_link_recovery_is_durable_before_the_only_target_move(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture = self._transaction_fixture(directory)
            with mock.patch.object(encoder, "ROOT", fixture["root"]), \
                    mock.patch.object(
                        encoder, "_transaction_lineage",  # noqa: SLF001
                        return_value=fixture["lineage"]):
                real_link = os.link

                def link_then_interrupt(source, destination, **kwargs):
                    real_link(source, destination, **kwargs)
                    raise OSError("synthetic post-link interruption")

                with self.assertRaisesRegex(OSError, "post-link"):
                    self._publish_transaction_receipt(
                        fixture, "PREPARED", fixture["prepared"],
                        link_side_effect=link_then_interrupt)
                paths = encoder._transaction_paths(  # noqa: SLF001
                    fixture["out"], fixture["contract"])
                events = []
                real_dir_fsync = encoder._fsync_directory  # noqa: SLF001

                def observed_dir_fsync(path):
                    events.append("directory-fsync")
                    return real_dir_fsync(path)

                def stop_before_move(**_kwargs):
                    self.assertFalse(paths["prepared_staged"].exists())
                    self.assertGreaterEqual(events.count("directory-fsync"), 2)
                    events.append("target-move")
                    raise OSError("synthetic before-move stop")

                with mock.patch.object(
                        encoder, "_fsync_directory",  # noqa: SLF001
                        side_effect=observed_dir_fsync), \
                        mock.patch.object(
                            encoder,
                            "_atomic_move_target_to_backup_no_replace",  # noqa: SLF001
                            side_effect=stop_before_move), \
                        self.assertRaisesRegex(OSError, "before-move"):
                    encoder._prepare_and_move_full_bank_v2_single_shard_transaction(  # noqa: E501, SLF001
                        out=fixture["out"], manifest={}, corpus_receipt={},
                        branch_smoke={}, contract_artifact={},
                        encoder_compute_dtype_correction_digest="0" * 64,
                        encoder_path_projection_correction_digest=
                            fixture["path_digest"],
                        transaction_contract=fixture["contract"],
                        transaction_contract_digest=fixture["contract_digest"])
                self.assertEqual(events[-1], "target-move")
                self.assertTrue(paths["prepared"].is_file())
                self.assertTrue(fixture["target"].is_file())

    def test_moved_resume_closes_durability_without_a_second_rename(self):
        for interrupted_fsync in (0, 1, 2):
            with self.subTest(interrupted_fsync=interrupted_fsync), \
                    tempfile.TemporaryDirectory() as directory:
                fixture = self._transaction_fixture(directory)
                with mock.patch.object(encoder, "ROOT", fixture["root"]), \
                        mock.patch.object(
                            encoder, "_transaction_lineage",  # noqa: SLF001
                            return_value=fixture["lineage"]):
                    self._publish_transaction_receipt(
                        fixture, "PREPARED", fixture["prepared"])
                    paths = encoder._transaction_paths(  # noqa: SLF001
                        fixture["out"], fixture["contract"])
                    # Model a process death immediately after the kernel's
                    # successful no-replace rename, before any durability
                    # close was guaranteed.
                    encoder._rename_noreplace(  # noqa: SLF001
                        fixture["target"], paths["backup"])
                    real_dir_fsync = encoder._fsync_directory  # noqa: SLF001
                    if interrupted_fsync:
                        fsync_count = 0

                        def interrupt_recovery(path):
                            nonlocal fsync_count
                            real_dir_fsync(path)
                            fsync_count += 1
                            if fsync_count == interrupted_fsync:
                                raise OSError(
                                    "synthetic durability-close interruption")

                        with mock.patch.object(
                                encoder, "_fsync_directory",  # noqa: SLF001
                                side_effect=interrupt_recovery), \
                                mock.patch.object(
                                    encoder, "_rename_noreplace",  # noqa: SLF001
                                    side_effect=AssertionError(
                                        "second rename attempted")), \
                                self.assertRaisesRegex(
                                    OSError, "durability-close"):
                            encoder._prepare_and_move_full_bank_v2_single_shard_transaction(  # noqa: E501, SLF001
                                out=fixture["out"], manifest={},
                                corpus_receipt={}, branch_smoke={},
                                contract_artifact={},
                                encoder_compute_dtype_correction_digest=
                                    "0" * 64,
                                encoder_path_projection_correction_digest=
                                    fixture["path_digest"],
                                transaction_contract=fixture["contract"],
                                transaction_contract_digest=
                                    fixture["contract_digest"])
                    events = []
                    real_file_fsync = (
                        encoder._fsync_exact_bound_regular_file)  # noqa: SLF001

                    def observed_file_fsync(path, binding, *, label):
                        events.append("backup-reopen-fsync")
                        return real_file_fsync(path, binding, label=label)

                    def observed_dir_fsync(path):
                        events.append(
                            "destination-dir-fsync"
                            if Path(path) == paths["backup"].parent
                            else "source-dir-fsync")
                        return real_dir_fsync(path)

                    with mock.patch.object(
                            encoder,
                            "_fsync_exact_bound_regular_file",  # noqa: SLF001
                            side_effect=observed_file_fsync), \
                            mock.patch.object(
                                encoder, "_fsync_directory",  # noqa: SLF001
                                side_effect=observed_dir_fsync), \
                            mock.patch.object(
                                encoder, "_rename_noreplace",  # noqa: SLF001
                                side_effect=AssertionError(
                                    "second rename attempted")):
                        status = (
                            encoder._prepare_and_move_full_bank_v2_single_shard_transaction(  # noqa: E501, SLF001
                                out=fixture["out"], manifest={},
                                corpus_receipt={}, branch_smoke={},
                                contract_artifact={},
                                encoder_compute_dtype_correction_digest=
                                    "0" * 64,
                                encoder_path_projection_correction_digest=
                                    fixture["path_digest"],
                                transaction_contract=fixture["contract"],
                                transaction_contract_digest=
                                    fixture["contract_digest"]))
                    self.assertEqual(
                        status["transaction_state"],
                        "MOVED_REGENERATION_PENDING")
                    self.assertEqual(events, [
                        "backup-reopen-fsync", "destination-dir-fsync",
                        "source-dir-fsync",
                    ])
                    self.assertFalse(fixture["target"].exists())
                    self.assertEqual(
                        paths["backup"].stat().st_ino,
                        fixture["target_binding"]["inode"])

    def test_complete_recomputes_live_stable_inventory_and_rejects_mutation(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture = self._transaction_fixture(directory)
            with mock.patch.object(encoder, "ROOT", fixture["root"]):
                context_records = [{
                    "path": "latents_v2/context/context.f16",
                    "sha256": "b" * 64, "byte_count": 1,
                    "shape": [3, 768, 1024],
                }]
                horizon_records = []
                for candidate_index in range(12):
                    path = (
                        "latents_v2/horizon/fixture-candidate-0.f16"
                        if candidate_index == 0 else
                        f"latents_v2/horizon/fixture-{candidate_index}.f16")
                    horizon_records.append({
                        "path": path, "sha256": f"{candidate_index:x}" * 64,
                        "byte_count": 1, "shape": [4, 768, 1024],
                    })
                latent_inventory = encoder._registered_latent_inventory(  # noqa: SLF001
                    fixture["out"], context_records + horizon_records)
                target_logical = fixture["target_binding"]["path"]
                stable_before = [{
                    "path": "immutable-stable.json",
                    "raw_sha256": "c" * 64, "byte_count": 10,
                }]
                fixture["evidence"].update({
                    "registered_smoke_shard_inventory_digest":
                        encoder.canonical_digest(latent_inventory),
                    "registered_smoke_non_target_shard_inventory_digest":
                        encoder._non_target_smoke_shard_inventory_digest(  # noqa: SLF001
                            latent_inventory, target_path=target_logical),
                    "registered_smoke_stable_artifact_inventory_digest":
                        encoder._stable_smoke_artifact_inventory_digest(  # noqa: SLF001
                            stable_before),
                })
                fixture["prepared"] = (
                    encoder.V2_DESIGN.
                    build_full_bank_v2_smoke_regeneration_prepared_receipt(
                        lineage=fixture["lineage"],
                        designated_target=fixture["target_binding"],
                        pretransaction_evidence=fixture["evidence"]))
                prepared_binding = self._publish_transaction_receipt(
                    fixture, "PREPARED", fixture["prepared"])
                paths = encoder._transaction_paths(  # noqa: SLF001
                    fixture["out"], fixture["contract"])
                encoder._atomic_move_target_to_backup_no_replace(  # noqa: SLF001
                    target_path=fixture["target"],
                    backup_path=paths["backup"],
                    target_binding=fixture["target_binding"])
                self._restore_transaction_target(fixture)
                smoke = self._protocol_smoke(fixture, prepared_binding)
                mutated_live = [{
                    "path": "immutable-stable.json",
                    "raw_sha256": "d" * 64, "byte_count": 10,
                }]
                durability_events = []
                real_file_fsync = (
                    encoder._fsync_exact_bound_regular_file)  # noqa: SLF001
                real_dir_fsync = encoder._fsync_directory  # noqa: SLF001

                def observed_file_fsync(path, binding, *, label):
                    durability_events.append(("file", Path(path), label))
                    return real_file_fsync(path, binding, label=label)

                def observed_dir_fsync(path):
                    durability_events.append(("directory", Path(path), ""))
                    return real_dir_fsync(path)

                with mock.patch.object(
                        encoder, "_registered_smoke_artifact_inventory",  # noqa: SLF001
                        return_value=mutated_live), \
                        mock.patch.object(
                            encoder,
                            "_non_target_smoke_shard_custody_inventory_digest",  # noqa: E501, SLF001
                            return_value=fixture["evidence"][
                                "registered_smoke_non_target_shard_custody_inventory_digest"]), \
                        mock.patch.object(
                            encoder, "_fsync_exact_bound_regular_file",  # noqa: SLF001
                            side_effect=observed_file_fsync), \
                        mock.patch.object(
                            encoder, "_fsync_directory",  # noqa: SLF001
                            side_effect=observed_dir_fsync), \
                        self.assertRaisesRegex(
                            RuntimeError, "registered stable artifact"):
                    encoder._complete_full_bank_v2_single_shard_transaction(  # noqa: E501, SLF001
                        out=fixture["out"], smoke=smoke,
                        index={"latents_index_digest": fixture["evidence"][
                            "latent_index_digest"]}, smoke_rows=[],
                        smoke_context_records=context_records,
                        smoke_horizon_records=horizon_records,
                        encoder_path_projection_correction_digest=
                            fixture["path_digest"],
                        transaction_contract=fixture["contract"],
                        transaction_contract_digest=fixture["contract_digest"],
                        expected_live_lineage=fixture["lineage"])
                self.assertFalse(paths["complete"].exists())
                self.assertEqual(durability_events[:2], [
                    ("file", fixture["target"],
                     "regenerated single-shard target"),
                    ("directory", fixture["target"].parent, ""),
                ])

    def test_path_projection_metadata_migration_is_exact_and_interrupt_safe(
            self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "repo"
            out = root / ".generated/go2_branch_corpus_v1_2/scorer_fit"
            out.mkdir(parents=True)
            context_records = []
            horizon_records = []
            inventory = []
            shard_paths = []
            for index in range(13):
                kind = "context" if index == 0 else "horizon"
                relative = Path("latents_v2") / kind / f"fixture-{index}.f16"
                path = out / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(f"immutable-shard-{index}".encode())
                path.chmod(0o640)
                digest = hashlib.sha256(path.read_bytes()).hexdigest()
                shape = [3, 1, 1] if index == 0 else [4, 1, 1]
                record = {
                    "path": str(relative), "sha256": digest,
                    "byte_count": path.stat().st_size, "shape": shape,
                }
                (context_records if index == 0 else horizon_records).append(
                    record)
                inventory.append({
                    **record,
                    "path": str(path.relative_to(root)),
                })
                shard_paths.append(path)

            historical_index = {
                "schema": encoder.FULL_BANK_V2_LATENT_INDEX_SCHEMA,
                "context_records": context_records,
                "horizon_records": horizon_records,
            }
            historical_index["latents_index_digest"] = (
                encoder.canonical_digest(historical_index))
            historical_smoke = {
                "schema": encoder.FULL_BANK_V2_SMOKE_SCHEMA,
                "latent_index_digest": historical_index[
                    "latents_index_digest"],
            }
            historical_smoke["smoke_receipt_digest"] = (
                encoder.canonical_digest(historical_smoke))
            index_path = out / encoder.FULL_BANK_V2_INDEX_NAME
            smoke_path = out / encoder.FULL_BANK_V2_SMOKE_NAME

            def raw(payload):
                return (json.dumps(payload, indent=2, sort_keys=True)
                        + "\n").encode()

            index_raw = raw(historical_index)
            smoke_raw = raw(historical_smoke)
            index_binding = {
                "path": str(index_path.relative_to(root)),
                "schema": historical_index["schema"],
                "self_digest_key": "latents_index_digest",
                "self_digest": historical_index["latents_index_digest"],
                "raw_sha256": hashlib.sha256(index_raw).hexdigest(),
                "byte_count": len(index_raw),
            }
            smoke_binding = {
                "path": str(smoke_path.relative_to(root)),
                "schema": historical_smoke["schema"],
                "self_digest_key": "smoke_receipt_digest",
                "self_digest": historical_smoke["smoke_receipt_digest"],
                "raw_sha256": hashlib.sha256(smoke_raw).hexdigest(),
                "byte_count": len(smoke_raw),
            }
            bundle = {
                "schema": (
                    "go2_scorer_fit_corpus_v2_path_projection_failure_"
                    "base_bundle_v1"),
                "latent_index_binding": index_binding,
                "base_smoke_receipt_binding": smoke_binding,
                "latent_shard_inventory": inventory,
                "context_latent_shard_count": 1,
                "horizon_latent_shard_count": 12,
                "total_latent_shard_count": 13,
                "total_latent_storage_bytes": sum(
                    item["byte_count"] for item in inventory),
            }
            bundle_digest = encoder.V2_DESIGN.canonical_digest(bundle)
            correction_digest = "f" * 64

            def reset_metadata():
                index_path.write_bytes(index_raw)
                smoke_path.write_bytes(smoke_raw)

            def custody():
                projection = []
                for path in shard_paths:
                    content = path.read_bytes()
                    metadata = path.stat()
                    projection.append((
                        content, metadata.st_dev, metadata.st_ino,
                        metadata.st_mode, metadata.st_nlink,
                        metadata.st_size, metadata.st_atime_ns,
                        metadata.st_mtime_ns, metadata.st_ctime_ns,
                    ))
                return projection

            with mock.patch.object(encoder, "ROOT", root):
                for interrupted_replace in (1, 2):
                    with self.subTest(interrupted_replace=interrupted_replace):
                        reset_metadata()
                        before = custody()
                        real_replace = os.replace
                        replace_count = 0

                        def replace_then_interrupt(source, target):
                            nonlocal replace_count
                            replace_count += 1
                            real_replace(source, target)
                            if replace_count == interrupted_replace:
                                raise OSError("synthetic interruption")

                        with mock.patch.object(
                                encoder.os, "replace",
                                side_effect=replace_then_interrupt):
                            with self.assertRaisesRegex(
                                    OSError, "synthetic interruption"):
                                encoder._migrate_historical_full_bank_v2_path_projection_metadata(  # noqa: E501, SLF001
                                    out=out, index_path=index_path,
                                    smoke_path=smoke_path,
                                    encoder_path_projection_correction_digest=
                                        correction_digest,
                                    base_smoke_artifact_bundle=bundle,
                                    base_smoke_artifact_bundle_digest=
                                        bundle_digest)
                        recovery_fsync_inodes = []
                        real_fsync = os.fsync

                        def observed_recovery_fsync(descriptor):
                            recovery_fsync_inodes.append(
                                os.fstat(descriptor).st_ino)
                            return real_fsync(descriptor)

                        with mock.patch.object(
                                encoder.os, "fsync",
                                side_effect=observed_recovery_fsync):
                            recovered = (
                                encoder._migrate_historical_full_bank_v2_path_projection_metadata(  # noqa: E501, SLF001
                                    out=out, index_path=index_path,
                                    smoke_path=smoke_path,
                                    encoder_path_projection_correction_digest=
                                        correction_digest,
                                    base_smoke_artifact_bundle=bundle,
                                    base_smoke_artifact_bundle_digest=
                                        bundle_digest))
                        if interrupted_replace == 2:
                            self.assertIn(
                                index_path.stat().st_ino,
                                recovery_fsync_inodes)
                            self.assertIn(
                                smoke_path.stat().st_ino,
                                recovery_fsync_inodes)
                            self.assertIn(
                                index_path.parent.stat().st_ino,
                                recovery_fsync_inodes)
                        self.assertEqual(recovered, interrupted_replace == 1)
                        migrated_index = json.loads(index_path.read_text())
                        migrated_smoke = json.loads(smoke_path.read_text())
                        self.assertEqual(
                            migrated_index[
                                encoder.ENCODER_PATH_PROJECTION_CORRECTION_DIGEST_FIELD],
                            correction_digest)
                        self.assertEqual(
                            migrated_smoke[
                                encoder.ENCODER_PATH_PROJECTION_CORRECTION_DIGEST_FIELD],
                            correction_digest)
                        self.assertEqual(
                            migrated_smoke["latent_index_digest"],
                            migrated_index["latents_index_digest"])
                        self.assertFalse(
                            encoder._migrate_historical_full_bank_v2_path_projection_metadata(  # noqa: E501, SLF001
                                out=out, index_path=index_path,
                                smoke_path=smoke_path,
                                encoder_path_projection_correction_digest=
                                    correction_digest,
                                base_smoke_artifact_bundle=bundle,
                                base_smoke_artifact_bundle_digest=
                                    bundle_digest))
                        self.assertEqual(custody(), before)

                reset_metadata()
                tampered_index = json.loads(index_path.read_text())
                tampered_index["unregistered"] = True
                tampered_index["latents_index_digest"] = (
                    encoder.canonical_digest({
                        key: value for key, value in tampered_index.items()
                        if key != "latents_index_digest"}))
                index_path.write_text(
                    json.dumps(tampered_index, indent=2, sort_keys=True) + "\n")
                with self.assertRaisesRegex(
                        RuntimeError, "historical raw binding changed"):
                    encoder._migrate_historical_full_bank_v2_path_projection_metadata(  # noqa: E501, SLF001
                        out=out, index_path=index_path, smoke_path=smoke_path,
                        encoder_path_projection_correction_digest=
                            correction_digest,
                        base_smoke_artifact_bundle=bundle,
                        base_smoke_artifact_bundle_digest=bundle_digest)

    def test_managed_alias_path_validates_physical_and_records_logical(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            repository = base / "repo"
            generated = repository / ".generated"
            generated.mkdir(parents=True)
            physical_root = base / "external/go2_branch_corpus_v1_2"
            physical_out = physical_root / "scorer_fit"
            relative = Path("latents_v2/horizon/fixture.f16")
            shard = physical_out / relative
            shard.parent.mkdir(parents=True)
            shard.write_bytes(b"synthetic-latent")
            alias = generated / "go2_branch_corpus_v1_2"
            alias.symlink_to(physical_root, target_is_directory=True)
            logical_out = alias / "scorer_fit"

            with mock.patch.object(encoder, "ROOT", repository):
                physical, logical = (
                    encoder._resolve_registered_logical_path(  # noqa: SLF001
                        logical_out, str(relative)))
                self.assertEqual(physical, shard.resolve())
                self.assertEqual(
                    logical,
                    Path(".generated/go2_branch_corpus_v1_2/scorer_fit")
                    / relative,
                )
                with self.assertRaisesRegex(
                        RuntimeError, "registered corpus path is not relative"):
                    encoder._resolve_registered_logical_path(  # noqa: SLF001
                        logical_out, "../escape.f16")

    def test_target_encoder_compute_dtype_is_fp32_on_cpu_and_cuda(self):
        self.assertEqual(
            encoder.TARGET_ENCODER_COMPUTE_DTYPE_NAME,
            "float32",
        )
        self.assertIs(
            encoder.target_encoder_compute_dtype(
                encoder.torch.device("cpu"), full_bank_v2=True),
            encoder.torch.float32,
        )
        self.assertIs(
            encoder.target_encoder_compute_dtype(
                encoder.torch.device("cuda:0"), full_bank_v2=True),
            encoder.torch.float32,
        )

    def test_legacy_compute_dtype_policy_is_unchanged_on_cpu_and_cuda(self):
        self.assertIs(
            encoder.target_encoder_compute_dtype(
                encoder.torch.device("cpu"), full_bank_v2=False),
            encoder.torch.float32,
        )
        self.assertIs(
            encoder.target_encoder_compute_dtype(
                encoder.torch.device("cuda:0"), full_bank_v2=False),
            encoder.torch.bfloat16,
        )

    def test_full_bank_dtype_binding_rejects_missing_or_bf16_lineage(self):
        encoder._validate_target_encoder_compute_dtype(  # noqa: SLF001
            "float32", label="fixture")
        for value in (None, "bfloat16"):
            with self.assertRaisesRegex(
                    RuntimeError, "target-encoder compute dtype changed"):
                encoder._validate_target_encoder_compute_dtype(  # noqa: SLF001
                    value, label="fixture")

    def test_dtype_correction_digest_binding_is_exact(self):
        digest = "e" * 64
        self.assertEqual(
            encoder._validate_encoder_compute_dtype_correction_digest(  # noqa: SLF001
                digest, label="fixture"),
            digest,
        )
        for value in (None, "e" * 63, "g" * 64):
            with self.assertRaisesRegex(
                    RuntimeError, "correction digest changed"):
                encoder._validate_encoder_compute_dtype_correction_digest(  # noqa: SLF001
                    value, label="fixture")

    def test_full_bank_v2_input_route_uses_only_exact_v2_producers(self):
        bindings = {key: "a" * 64
                    for key in encoder.FULL_BANK_V2_BINDING_KEYS}
        bindings["scorer_fit_corpus_v2_scorer_contract_digest"] = "c" * 64
        bindings[
            "scorer_fit_corpus_v2_scorer_contract_artifact_digest"] = "d" * 64
        bindings["target_encoder_checkpoint_sha256"] = (
            encoder.contract()["target_encoder"]["checkpoint_sha256"])
        manifest = {
            "schema":
                encoder.CORPUS_BUILDER.SCORER_FIT_V2_STATE_MANIFEST_SCHEMA,
            "pool": "scorer_fit_v2",
            "state_manifest_digest": "b" * 64,
            "attempted_branch_count_registered": 1_440,
            "states": [
                {"candidate_indices": list(range(12))}
                for _ in range(120)
            ],
            **bindings,
        }
        rows = []
        for candidate_index in range(12):
            row = {
                "state_id": "smoke-state",
                "candidate": f"candidate-{candidate_index}",
                "candidate_index": candidate_index,
                "valid": True,
                "state_manifest_digest": manifest["state_manifest_digest"],
                **bindings,
            }
            row["branch_row_digest"] = encoder.canonical_digest(row)
            rows.append(row)
        successor = {
            "preoutcome_lineage": {
                "scorer_fit_corpus_v2_source_correction_digest": bindings[
                    "scorer_fit_corpus_v2_source_correction_digest"],
            },
            "state_selector_binding": {
                "state_manifest_digest": manifest["state_manifest_digest"],
                "assignment_manifest_digest": manifest[
                    "full_bank_assignment_manifest_digest"],
            },
            "protected_predecessor_scientific_contract": {
                "target_encoder": encoder.contract()["target_encoder"],
            },
            encoder.V2_CONTRACT.CONTRACT_SELF_KEY: "c" * 64,
        }
        artifact = {
            "contract": successor,
            encoder.V2_CONTRACT.CONTRACT_SELF_KEY: "c" * 64,
            encoder.V2_CONTRACT.ARTIFACT_SELF_KEY: "d" * 64,
        }
        dtype_correction = {
            encoder.V2_DESIGN.ENCODER_COMPUTE_DTYPE_CORRECTION_SELF_KEY:
                "e" * 64,
        }
        immutable_dtype = {"payload": dtype_correction, "binding": {}}
        base_bundle = {"fixture": "authority-bound-base-smoke"}
        transaction_contract = (
            encoder.V2_DESIGN.
            ENCODER_PATH_PROJECTION_SINGLE_SHARD_REGENERATION_TRANSACTION_CONTRACT)
        path_correction = {
            encoder.V2_DESIGN.ENCODER_PATH_PROJECTION_CORRECTION_SELF_KEY:
                "f" * 64,
            "immutable_encoder_compute_dtype_correction": immutable_dtype,
            "immutable_encoder_compute_dtype_correction_digest": "e" * 64,
            "immutable_base_smoke_artifact_bundle": base_bundle,
            "base_smoke_artifact_bundle_digest":
                encoder.V2_DESIGN.canonical_digest(base_bundle),
            "single_shard_regeneration_transaction_contract":
                transaction_contract,
            "single_shard_regeneration_transaction_contract_digest":
                encoder.V2_DESIGN.canonical_digest(transaction_contract),
        }
        bundle = {
            "manifest": manifest,
            "receipt": {"complete": False},
            "rows": rows,
            "scorer_contract": artifact,
        }
        with mock.patch.object(
                encoder.V2_DESIGN, "load_active_design_authority",
                return_value={
                    "source_correction_digest": bindings[
                        "scorer_fit_corpus_v2_source_correction_digest"],
                    "encoder_compute_dtype_correction_digest": "e" * 64,
                    "encoder_compute_dtype_correction": dtype_correction,
                    "encoder_path_projection_correction_digest": "f" * 64,
                    "encoder_path_projection_correction": path_correction,
                }), \
                mock.patch.object(
                    encoder.V2_DESIGN,
                    "validate_immutable_encoder_compute_dtype_correction",
                    return_value=immutable_dtype), \
                mock.patch.object(
                encoder.CORPUS_BUILDER,
                "load_and_validate_full_bank_v2_branch_outputs_for_consumption",
                return_value=bundle) as producer, \
                mock.patch.object(
                    encoder.V2_CONTRACT, "load_contract_for_consumption",
                    return_value=artifact) as contract_loader, \
                mock.patch.object(
                    encoder.V2_CONTRACT, "validate_contract_artifact",
                    return_value=artifact), \
                mock.patch.object(
                    encoder.ALLOC, "allocation_contract_digest",
                    side_effect=AssertionError("legacy ALLOC opened")) as alloc, \
                mock.patch.object(
                    encoder, "_load_inputs",
                    side_effect=AssertionError("legacy route opened")) as legacy:
            observed = encoder._load_full_bank_v2_inputs(
                encoder.OUT_ROOT / "scorer_fit", allow_partial=True)
        producer.assert_called_once_with(
            out=encoder.OUT_ROOT / "scorer_fit", allow_partial=True)
        contract_loader.assert_called_once_with(
            root=encoder.ROOT,
            encoder_path_projection_correction=path_correction)
        alloc.assert_not_called()
        legacy.assert_not_called()
        self.assertEqual(observed[0], manifest)
        self.assertEqual(len(observed[2]), 12)
        self.assertEqual(observed[4], "e" * 64)
        self.assertEqual(observed[5], "f" * 64)
        self.assertEqual(observed[6], base_bundle)
        self.assertEqual(
            observed[7], encoder.V2_DESIGN.canonical_digest(base_bundle))
        self.assertEqual(observed[8], transaction_contract)
        self.assertEqual(
            observed[9], encoder.V2_DESIGN.canonical_digest(
                transaction_contract))

    def test_full_bank_v2_source_correction_mismatch_precedes_branch_producer(
            self):
        artifact = {
            "contract": {
                "preoutcome_lineage": {
                    "scorer_fit_corpus_v2_source_correction_digest": "b" * 64,
                },
            },
        }
        dtype_correction = {
            encoder.V2_DESIGN.ENCODER_COMPUTE_DTYPE_CORRECTION_SELF_KEY:
                "e" * 64,
        }
        immutable_dtype = {"payload": dtype_correction, "binding": {}}
        base_bundle = {"fixture": "authority-bound-base-smoke"}
        path_correction = {
            encoder.V2_DESIGN.ENCODER_PATH_PROJECTION_CORRECTION_SELF_KEY:
                "f" * 64,
            "immutable_encoder_compute_dtype_correction": immutable_dtype,
            "immutable_encoder_compute_dtype_correction_digest": "e" * 64,
            "immutable_base_smoke_artifact_bundle": base_bundle,
            "base_smoke_artifact_bundle_digest":
                encoder.V2_DESIGN.canonical_digest(base_bundle),
        }
        with mock.patch.object(
                encoder.V2_DESIGN, "load_active_design_authority",
                return_value={
                    "source_correction_digest": "a" * 64,
                    "encoder_compute_dtype_correction_digest": "e" * 64,
                    "encoder_compute_dtype_correction": dtype_correction,
                    "encoder_path_projection_correction_digest": "f" * 64,
                    "encoder_path_projection_correction": path_correction,
                }), \
                mock.patch.object(
                    encoder.V2_DESIGN,
                    "validate_immutable_encoder_compute_dtype_correction",
                    return_value=immutable_dtype), \
                mock.patch.object(
                    encoder.V2_CONTRACT, "load_contract_for_consumption",
                    return_value=artifact), \
                mock.patch.object(
                    encoder.V2_CONTRACT, "validate_contract_artifact",
                    return_value=artifact), \
                mock.patch.object(
                    encoder.CORPUS_BUILDER,
                    "load_and_validate_full_bank_v2_branch_outputs_for_consumption",
                    side_effect=AssertionError("branch producer opened")) as producer:
            with self.assertRaisesRegex(
                    RuntimeError, "source-correction lineage changed"):
                encoder._load_full_bank_v2_inputs(
                    encoder.OUT_ROOT / "scorer_fit", allow_partial=True)
        producer.assert_not_called()

    def test_missing_path_correction_precedes_contract_and_branch_producer(
            self):
        with mock.patch.object(
                encoder.V2_DESIGN, "load_active_design_authority",
                return_value={"source_correction_digest": "a" * 64}), \
                mock.patch.object(
                    encoder.V2_CONTRACT, "load_contract_for_consumption",
                    side_effect=AssertionError("contract opened")) as contract, \
                mock.patch.object(
                    encoder.CORPUS_BUILDER,
                    "load_and_validate_full_bank_v2_branch_outputs_for_consumption",
                    side_effect=AssertionError("branch producer opened")) as producer:
            with self.assertRaisesRegex(
                    RuntimeError,
                    "active encoder-path-projection correction is unavailable"):
                encoder._load_full_bank_v2_inputs(  # noqa: SLF001
                    encoder.OUT_ROOT / "scorer_fit", allow_partial=True)
        contract.assert_not_called()
        producer.assert_not_called()

    def test_full_bank_v2_output_registry_is_versioned_and_disjoint(self):
        self.assertEqual(encoder.FULL_BANK_V2_INDEX_NAME,
                         "latents_index_v2.json")
        self.assertEqual(encoder.FULL_BANK_V2_SMOKE_NAME,
                         "smoke_encoding_receipt_v2.json")
        self.assertEqual(encoder.FULL_BANK_V2_LATENTS_NAME, "latents_v2")
        self.assertNotIn("candidate_allocator_contract_digest",
                         encoder.FULL_BANK_V2_BINDING_KEYS)
        self.assertIn("scorer_fit_corpus_v2_source_correction_digest",
                      encoder.FULL_BANK_V2_BINDING_KEYS)

    def test_branch_row_remains_bound_to_manifest_scientific_contract(self):
        historical = "1" * 64
        manifest = {
            "state_manifest_digest": "2" * 64,
            **{key: "3" * 64 for key in encoder.CORPUS_BINDING_KEYS},
            "scorer_contract_v1_2_digest": historical,
        }
        row = {
            "state_id": "fixture-state",
            "candidate": "fixture-candidate",
            "state_manifest_digest": manifest["state_manifest_digest"],
            **{key: manifest[key] for key in encoder.CORPUS_BINDING_KEYS},
        }
        row["branch_row_digest"] = encoder.canonical_digest(row)
        encoder._validate_row(row, manifest, historical)
        with self.assertRaisesRegex(RuntimeError, "scorer_contract_v1_2_digest"):
            encoder._validate_row(row, manifest, encoder.contract_digest())

    def test_global_exact_manifest_uses_successor_operational_contract(self):
        historical_contract = "0" * 64
        current_contract = encoder.contract_digest()
        predecessor = {
            "clean_source_launch_receipt_digest": "1" * 64,
            "source_repository_commit": "2" * 40,
            "clean_source_binding_digest": "3" * 64,
            "bound_implementations_digest": "4" * 64,
            "scorer_contract_artifact_digest": "5" * 64,
        }
        successor = {
            "clean_source_launch_receipt_digest": "a" * 64,
            "source_repository_commit": "b" * 40,
            "clean_source_binding_digest": "c" * 64,
            "bound_implementations_digest": "d" * 64,
            "scorer_contract_artifact_digest": "e" * 64,
            "clean_source_launch_receipt_sha256": "6" * 64,
            "scorer_contract_artifact_sha256": "7" * 64,
            "launch_state_selector_feasibility_receipt_digest": "8" * 64,
            "mixed_precontract_disposition_receipt_digest": "9" * 64,
            "global_exact_execution_amendment_digest": "a" * 64,
            "global_exact_successor_scorer_contract_digest": "b" * 64,
            "current_scorer_contract_v1_2_digest": current_contract,
            "scientific_predecessor_launch_bindings": predecessor,
        }
        manifest = {
            "small_completion_global_exact_execution": {},
            **predecessor,
            "mixed_precontract_disposition_receipt_digest": "9" * 64,
            "scorer_contract_v1_2_digest": historical_contract,
        }
        with mock.patch.object(
                encoder.CORPUS_BUILDER,
                "load_global_exact_successor_scorer_contract_for_consumption",
                return_value=successor, create=True) as load_successor, \
                mock.patch.object(
                    encoder, "_load_clean_source_launch_receipt") as legacy:
            operational, scientific, selector = (
                encoder._load_manifest_launch_lineage(manifest))
        load_successor.assert_called_once_with(manifest)
        legacy.assert_not_called()
        self.assertEqual(
            operational["current_scorer_contract_v1_2_digest"],
            current_contract)
        self.assertEqual(
            operational["global_exact_scorer_contract_lineage"], {
                "schema":
                    encoder.GLOBAL_EXACT_SCORER_CONTRACT_LINEAGE_SCHEMA,
                "scientific_predecessor_scorer_contract_v1_2_digest":
                    historical_contract,
                "current_scorer_contract_v1_2_digest": current_contract,
                "global_exact_successor_scorer_contract_digest": "b" * 64,
            })
        self.assertEqual(
            {key: scientific[key]
             for key in encoder.SCIENTIFIC_PREDECESSOR_LAUNCH_BINDING_KEYS},
            predecessor)
        self.assertEqual(
            selector["launch_state_selector_feasibility_receipt_digest"],
            "8" * 64)

        malformed = dict(
            operational["global_exact_scorer_contract_lineage"])
        malformed["unexpected"] = "f" * 64
        with self.assertRaisesRegex(RuntimeError, "schema is not closed"):
            encoder._validate_global_exact_scorer_contract_lineage(malformed)

    def test_live_selection_replay_failure_precedes_rows_and_frames(self):
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory) / "scorer_fit"
            out.mkdir()
            manifest = {"pool": "scorer_fit", "states": []}
            (out / "state_manifest.json").write_text(json.dumps(manifest))
            with mock.patch.object(
                    encoder.CORPUS_BUILDER,
                    "load_active_state_manifest_for_consumption",
                    side_effect=RuntimeError(
                        "later replacement capture prefix is not canonical"
                    )) as replay, \
                    mock.patch.object(
                        encoder, "_load_selector_successor_receipts"
                    ) as selector, \
                    mock.patch.object(encoder, "_verify_frames") as frames, \
                    mock.patch.object(encoder.json, "loads") as raw_json:
                with self.assertRaisesRegex(
                        RuntimeError, "later replacement capture prefix"):
                    encoder._load_inputs(
                        out, allow_partial=False, pool="scorer_fit")
            replay.assert_called_once_with(
                out / "state_manifest.json", pool="scorer_fit")
            selector.assert_not_called()
            frames.assert_not_called()
            raw_json.assert_not_called()

    def test_global_selector_replay_never_calls_legacy_allocation_validator(self):
        feasibility = {
            "state_selector_feasibility_receipt_digest": "f" * 64}
        disposition = {
            "mixed_precontract_disposition_receipt_digest": "d" * 64}
        revalidation = {
            "preserved_state_revalidation_receipt_digest": "e" * 64}
        allocation = {"allocation_manifest_digest": "1" * 64}
        manifest = {"small_completion_global_exact_execution": {}}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            out_root = root / ".generated/go2_branch_corpus_v1_2"
            scorer_fit = out_root / "scorer_fit"
            scorer_fit.mkdir(parents=True)
            (root / encoder.STATE_SELECTOR.
             STATE_SELECTOR_FEASIBILITY_RECEIPT_PATH).parent.mkdir(
                 parents=True, exist_ok=True)
            (root / encoder.STATE_SELECTOR.
             STATE_SELECTOR_FEASIBILITY_RECEIPT_PATH).write_text(
                 json.dumps(feasibility))
            (root / encoder.STATE_SELECTOR.
             PRESERVED_STATE_MIXED_PRECONTRACT_DISPOSITION_RECEIPT_PATH
             ).write_text(json.dumps(disposition))
            (root / encoder.STATE_SELECTOR.
             PRESERVED_STATE_REVALIDATION_RECEIPT_PATH).write_text(
                 json.dumps(revalidation))
            (scorer_fit / "candidate_allocation_manifest.json").write_text(
                json.dumps(allocation))
            with mock.patch.object(encoder, "ROOT", root), \
                    mock.patch.object(encoder, "OUT_ROOT", out_root), \
                    mock.patch.object(
                        encoder.STATE_SELECTOR,
                        "validate_authority_artifacts"), \
                    mock.patch.object(
                        encoder.STATE_SELECTOR,
                        "validate_frozen_reachability_feasibility_pass",
                        return_value=feasibility), \
                    mock.patch.object(
                        encoder.STATE_SELECTOR,
                        "validate_preserved_state_mixed_precontract_disposition_receipt"), \
                    mock.patch.object(
                        encoder.STATE_SELECTOR,
                        "validate_preserved_state_revalidation_receipt") as legacy, \
                    mock.patch.object(
                        encoder.CORPUS_BUILDER,
                        "validate_global_exact_allocation_for_consumption",
                        return_value={
                            "preserved_state_revalidation_receipt_digest":
                                "e" * 64,
                        }) as certify:
                bindings = encoder._load_selector_successor_receipts(
                    source_commit="c" * 40,
                    selection_digest="b" * 64,
                    active_states=[],
                    expected_feasibility_receipt_digest="f" * 64,
                    expected_mixed_precontract_disposition_receipt_digest=
                        "d" * 64,
                    expected_clean_source_binding_digest="a" * 64,
                    expected_bound_implementations_digest="9" * 64,
                    global_exact_manifest=manifest)
            certify.assert_called_once_with(manifest, allocation)
            legacy.assert_not_called()
            self.assertEqual(
                bindings["preserved_state_revalidation_receipt_digest"],
                "e" * 64)


if __name__ == "__main__":
    unittest.main()
