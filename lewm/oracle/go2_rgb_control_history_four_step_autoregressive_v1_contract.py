"""Frozen contract for the bounded RGB four-step rollout-objective assay.

This development-only experiment asks whether direct H1--H4 autoregressive
supervision improves future-latent fidelity and action-specific branch
discrimination relative to the already completed two-step objective.  It is
predictive-dynamics work only: no utility scorer, selected-action endpoint,
planning claim, simulator-corpus generation, or sealed material is authorised.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
import subprocess
from typing import Any, Mapping


ROOT = Path(__file__).resolve().parents[2]
STATUS = "DEVELOPMENT_ONLY_NOT_CLAIM_BEARING"
LABEL = "EXPLORATORY_RGB_CONTROL_HISTORY_FOUR_STEP_AUTOREGRESSIVE_V1"
CONTRACT_SCHEMA = "go2_rgb_control_history_four_step_autoregressive_v1_contract_v1"
CONTRACT_SELF_KEY = "contract_digest"
SOURCE_SCHEMA = "go2_rgb_control_history_four_step_autoregressive_v1_source_closure_v1"
SOURCE_SELF_KEY = "four_step_source_closure_digest"
BASE_SOURCE_COMMIT = "b84f03a9e270f1d1eb3b5b0e12c2a03d711f00f9"

NEW_SOURCE_PATHS = (
    "lewm/oracle/go2_rgb_control_history_four_step_autoregressive_v1_contract.py",
    "scripts/run_go2_rgb_control_history_four_step_autoregressive_v1.py",
    "lewm/tests/test_go2_rgb_control_history_four_step_autoregressive_v1_contract.py",
    "lewm/tests/test_run_go2_rgb_control_history_four_step_autoregressive_v1.py",
)

FROZEN_SOURCE_FILES = {
    "scripts/dev_checkpoint_v1.py":
        ("6965aeb907edadbe40128be64b6725cbbcb3f0c963da8c825ab854db9e90d860", 7_935),
    "scripts/dev_proprio_predictor_v1.py":
        ("04e3b140727f3c3c661416940cca40bcdc3925d943c3c90628516e35da43ada0", 15_179),
    "scripts/run_dev_proprio_factorial_driver_v1.py":
        ("06cd749d23605469bbce8d9119c94180a6a8814a34680ff04e6e86b1ac397031", 33_794),
    "scripts/run_dev_v03_temporal_action_jepa_v1.py":
        ("06e92ba7301ef710a68c4e16645f38951ccc2d10b07d97bb7dc1fdbda948c949", 16_127),
    "scripts/dev_action_slew_reconstruction_v1.py":
        ("17075cc10bdfc637a630da1b495f064156be9d481b6be631f50fd1e370b9203e", 13_469),
    "scripts/build_dev_v03_proprio_action_manifest_v1.py":
        ("27f65c0ead7ccc444347301b8c58b74cd877343436271fa85b95ebc1dc744b81", 23_046),
    "scripts/freeze_dev_proprio_run_package_v1.py":
        ("b555a2c3b380142421ba83aa5f41a38aedb0d25077d4ae4c53ffcb5169b8deca", 6_704),
    "scripts/build_dev_factorial_manifest_v1.py":
        ("46d12c8f0bd9e1b8e04245e2cadd3a9f7dad30af1be8cca2445f3b0db4d3ae48", 9_775),
    "scripts/build_dev_v03_horizon_sequences_v1.py":
        ("1e4860bddc07b3df6a1990b57634130643333dee42935e5ef32939a30a54e1e8", 6_913),
    "scripts/eval_dev_v03_horizon_rollout_v1.py":
        ("634146e62c396f024a90158f19493208c602cb48d64cc9dbe0064246795b337d", 13_371),
    "scripts/eval_dev_proprio_factorial_v1.py":
        ("f3513bac72350cf9abdf1e31f80dd0ea3119595450f9ee7336ae94b5e70c08a3", 10_897),
    "scripts/analyze_go2_counterfactual_predictor_qualification_v1_2.py":
        ("63c8502d5bba9bd49b8adfe2b480ecc567b764e359aa296baceafdefa5ba8bea", 128_772),
    "scripts/run_go2_counterfactual_occupancy_assay_v1_2.py":
        ("2b1fe088105054cf691e74960c21c30332d1e33ee71e42340c5ff8afb23d50ee", 136_102),
    "scripts/build_dev_canonical_cache_map_v1.py":
        ("6d6212bb842b386093fcac69c2f868f7ab887e994b950b51d73b27c5741a0753", 11_867),
    "scripts/dev_frozen_dense_representation_encoders_v1.py":
        ("c5bb12ddc4711071dbdbac8c2ad6cc4b7528dd8ceb263b752fd539bd954aa9e2", 15_564),
    "scripts/run_dev_v03_two_step_rollout_v1.py":
        ("03c2621c9d11d5741a5587ae982b20de140182f0e68d5c33027486cd48b47879", 17_870),
    "scripts/dev_proprio_experiment_config_v1.py":
        ("97ebcdeb8ba263c3b863284e92de5a26f3074c1533b87d204543ce1b857f9105", 12_498),
    "lewm/benchmarks/go2_dynamic_cell_square_projection.py":
        ("ce2bb0d38ed1436635cdd1468ba1dfe1a935fdafdd6dda5adcf37b97a32a74bf", 9_633),
    "lewm/benchmarks/go2_observable_camera_ray_evidence_v4.py":
        ("708d368e461fe60aacb860dda5b0cbfd1acaf43e5cb3ae18a77bb48de739fb85", 23_757),
    "lewm/oracle/go2_textured_v03_renderer.py":
        ("392439be92c128f639c8c9682627530b34660168229c24c9944d847372524aba", 15_420),
    "lewm_worlds/lewm_worlds/manifest.py":
        ("5679768016226e89e385ec7a7238616416248a9a1194b898ecb9078662f6a888", 12_154),
    "scripts/build_go2_observable_camera_ray_fit_v4.py":
        ("4efb0517130df39a1953539755d82289b16e89b314bba5713d6d9d944acf1d16", 69_970),
    "scripts/run_dev_frozen_dense_representation_screen_v1.py":
        ("6402883b211f7cb40a923e78e9ba78c9510bb0310b25df0c60ce9b73cba530cb", 32_943),
    "scripts/run_go2_representation_qualification_probe_v1.py":
        ("75ddc9e7674549e385a56f6866e9c2a39c034512f9b48af85cb3acb937c75b9a", 15_427),
    "scripts/dev_seed_reestimation_v1.py":
        ("524c7a0bec20c22bbf313b8585b16fac278c2de21331a83a16808f72d2d03536", 5_900),
}

CACHE_ROOT = Path("/home/andrewknowles/.cache/lewm_go2_temporal_v03")
FACTORIAL_ROOT = CACHE_ROOT / "factorial_v1"
RUN_PACKAGE_PATH = CACHE_ROOT / "proprio_v1/scientific_run_package.json"
REGISTERED_RUNTIME_PARENT = CACHE_ROOT
RUNTIME_RELATIVE = Path("four_step_rollout_v1")
CONTRACT_RELATIVE = RUNTIME_RELATIVE / "contract.json"

FROZEN_SEEDS = (
    2_026_080_901, 2_026_080_902, 2_026_080_903, 2_026_080_904,
    2_026_080_905, 2_026_080_906, 2_026_080_907, 2_026_080_908,
)
FAMILIES = (
    "large_enclosed_maze",
    "local_composite_motifs",
    "loop_alias_stress",
    "medium_enclosed_maze",
    "open_obstacle_field",
    "rough_local_dynamics",
    "small_enclosed_maze",
    "visual_sensor_stress",
)

BASE_WEIGHT_SHA256 = {
    2_026_080_901: "afa42cd84d6ff86fdd40996c23d23659627a2f23a498c3d77dc0ccbd2e0cf5ed",
    2_026_080_902: "3a87efa35d296044efa278f6e6337555e26f4aa608a49f16e1e8afa705c1bf01",
    2_026_080_903: "b4b50aa0b376925fa29a72ceb723edd1f1c62c32427dece89b7f7c6a0b06beb4",
    2_026_080_904: "26433af691517910f57f2acbaae8ddc9e731f02545bf9a40506e0407c203054f",
    2_026_080_905: "6f8f9ebd45fa280e341f107ec6025fa545ead41097996e4f90f5a13331e675af",
    2_026_080_906: "290751d3b380255212ee48c1d8631de35771cb289d0e7be32165015a0d0a77ae",
    2_026_080_907: "4e14df50f3efd61b4f4fddb1a4b2aa52c2f89b0e2ef1178ab3eb584345ac6fdf",
    2_026_080_908: "d28783bfd50ebbeb2b4a97e6a7a7d897bcdee01f5feaf13d491cd1d260a8c73a",
}
BASE_STATE_DIGEST = {
    2_026_080_901: "29f5cb3995ca966b814e4cb09c3f0ef103756beeb12c169299ec6afcbee3dcd8",
    2_026_080_902: "8ce14bde86e29945ab2d1e4a3cc4a276f34b07a1101252dc5084286b8e5fec02",
    2_026_080_903: "da7fd8331151a2d8528d9e3664014956c2be4c01939788eab94a723cef2a7c23",
    2_026_080_904: "feb6785995b471c1e53a03b3d57bc914f2d5b407a83d3c3f43e9840e9e9006db",
    2_026_080_905: "6814c8355b4a076ce4edaf470af066b500795ad5ced73f7d8f343f7a6b9a0414",
    2_026_080_906: "4da0f438f7463dfe73deb098ccd6ef2af84aaf0c1dd64d147c63cef09eaf3143",
    2_026_080_907: "a9993820c158ecaf95f5fad0c7af2c0bbfbbd8a2d3c24ef236baf999afcb5a9d",
    2_026_080_908: "6bca23d0f511d88bcc2fff7edab69ce638b507828ba050f5ba6c175383b5529c",
}
BASE_WEIGHT_BYTES = 68_832_569
COMPARATOR_CHECKPOINT_BYTES = {
    seed: (206_534_615 if seed in (2_026_080_907, 2_026_080_908)
           else 206_534_551)
    for seed in FROZEN_SEEDS
}

COMPARATOR_CHECKPOINT_SHA256 = {
    2_026_080_901: {
        "rgb_one_step": "20b6e3fa2a2d3c3ec2c20ea37e524f9c2872fdcfd5226b114822efa26872261a",
        "rgb_two_step_rollout": "75e7a8f5eb5416100dd91fdd07c6aeae1c8fa2255ef189bfde2a5ce300f881b4",
    },
    2_026_080_902: {
        "rgb_one_step": "085702386da4b36bafe6ff432ca955a2b1a9a69de9a8023aa4fc3b099953f0ff",
        "rgb_two_step_rollout": "90bbf9a8117dbf528d9693415becd5c9e9605ecad02520f3e00513dfee691530",
    },
    2_026_080_903: {
        "rgb_one_step": "a7878c6159cceae8f69f84927bd1ee3a4c3d8dbf6d1e97003eb9ebdae1f91bc4",
        "rgb_two_step_rollout": "b769ef91f1ef17377f7c7f184c85ea0a9859ead2b87aa8351a89b7a05192aad1",
    },
    2_026_080_904: {
        "rgb_one_step": "1386b6303ac5b47fea7a67e831a375d164ba372ee4bf60fd87609ed35352d1ff",
        "rgb_two_step_rollout": "aad6711b6d15e6664038ace1fe0f376516256062c2235334b74bfb68135e419a",
    },
    2_026_080_905: {
        "rgb_one_step": "5d78f18e0d0052479cb81a43acbaa953bebeb6fc13dac58c506211c46416a1e9",
        "rgb_two_step_rollout": "c474a5b09c041aa263950b3b2b8bd2369d3644aec7019268610fea4b846b6386",
    },
    2_026_080_906: {
        "rgb_one_step": "846fbe05f78e9b513841cb08f71858e9fdb7dd4430181bca140d29f72574a200",
        "rgb_two_step_rollout": "fc480799cc637f5c3d4bd582da233e38b76d422b48833075e018d49df517aa1a",
    },
    2_026_080_907: {
        "rgb_one_step": "86d9f6108f40b8d2cf49e5264fc998412493258258b86391067c71193066afbc",
        "rgb_two_step_rollout": "4501841125eee43568e6031d4061d23b309c080f11b129538dadb6cfc8a05432",
    },
    2_026_080_908: {
        "rgb_one_step": "025aff4d9bc7380b4a51e4ac08282bbeafb2be189bf27d31d48bcf247f2b02f2",
        "rgb_two_step_rollout": "a39f5050c02ab7b002c6b1c76256dc2b5783046cf5b877cc6d5354880c45b89a",
    },
}
COMPARATOR_OBJECTIVES = {
    "rgb_one_step": "L1",
    "rgb_two_step_rollout": "1.5 * L1 + 0.5 * L2",
    "note": (
        "historical frozen objectives are not rescaled, retrained, or "
        "reinterpreted; the new four-step average is the registered new arm"),
}

FROZEN_FACTORIAL = {
    "confirmatory_commit": "443e5914694a533534486b629e95ec15f8df9b7a",
    "final_report_digest": "60b0bb2d0b13ba47eac5e306c33d97dcfdce31102870edfc50b01f7f9b247161",
    "final_report_file_sha256": "288e830625842746089608f2a26affa90e0428c3c831c5da5cc5969972fc2455",
    "final_report_bytes": 27_047,
    "run_package_digest": "cf0456bef0cbe7cd8f2cd666b600f91ebf845f6156d180569edf36be53552991",
    "run_package_file_sha256": "45e52cee60641c21fe05ec875e1941aa086ef1b29dc6f3528d96197cf62fdfd4",
    "run_package_bytes": 7_656,
    "factorial_manifest_digest": "6ff053033475debd3d8bb415080efb15adfaefc31f01295b956bd85c12b6dac0",
    "factorial_manifest_file_sha256": "8bf59020d24e02fdb11948f3732220df839aa1c3bc8612392ce6baab6b8d629c",
    "factorial_manifest_bytes": 2_305_678,
    "proprio_rows_file_sha256": "7b79d12830f12175c591a87982a20e5df7a8d64cfc40e99dd9cee2dc1ae2543e",
    "canonical_cache_map_digest": "a45bcc7d46da3c085f0603e79e568f1228b76c489868d6a96aed2b1485d85a7e",
    "model_configuration_sha256":
        "582e7088c2230963fa9b5a0acde4e3de0a863d4c55af74dd7c53d5c1eb18497a",
    "normalisation_sha256": "f5ea58b29d79362d4d814ff1b4225b54a5c97fb95442c866def80b0c2c4c2fab",
    "seed_registry_sha256": "bbaffee2f246813778e7c7195794414541dc9d298b6877df8562f359f21ba3a6",
    "seed_registry_file_sha256": "75f9934418679f9c0d62e0ad0abc38faf77b2d21c1a78505c0b3a0b8f8b41974",
    "seed_registry_bytes": 8_028,
    "seed_execution_audit_digest":
        "03ad9d3bd588b251385240d598cd1915344f97437d1f8e8f8b33705cc86760f0",
    "seed_execution_audit_file_sha256":
        "c5f99c5570c02cd8b3a5f6ede14d5928e14b9ee5047a49ebd77590341ef8ec0c",
    "seed_execution_audit_bytes": 39_449,
    "initial_launch_receipt_digest": "abe036ad3044467496ee1ead5cedef8ed40362220e841f23f7e443b45274a4fa",
    "initial_launch_receipt_file_sha256": "fe3786d9ac7bb23fca776aa02a64e9dc7801962d5f28d635c34110056ccd4f85",
    "initial_launch_receipt_bytes": 2_234,
    "continuation_receipt_digest": "5f3378955a145fee342a7c9bb313b60e8d8aa7924770ca0d20bb3f06e6c51e4c",
    "continuation_receipt_file_sha256": "53bf2948d79cfff0fa294c2872ddd3cf56ba51308d5b3da875de7f10228ce439",
    "continuation_receipt_bytes": 6_598,
    "historical_train_rows": 3_922,
    "checkpoint_epoch": 21,
    "historical_cells_retrained_or_reselected": False,
}

CLOSED_SCIENTIFIC_LINES = {
    "fixed_pooling_ViT_L": {
        "source_lineage": "20aa87496f237b0769486d3e558e833bd6aa03ab",
        "safety_auc": 0.7043234199,
        "latent_over_baseline_pairwise_gain": 0.0317880795,
        "terminal": "VALID_SCIENTIFIC_QUALIFICATION_FAILURE",
        "qualified_scorer_package": False,
        "predictor_utility_scoring": False,
    },
    "ViT_g_scale_ablation": {
        "source_commit": "8d36aeea09d1dc069d53dfb48675da560ea0c343",
        "result_digest": "b8b98bb7f5ae607d023a20876107cead59c3bdfa0a858955ea0d760ea5973f0a",
        "safety_auc": 0.6332379770,
        "latent_over_baseline_pairwise_gain": 0.0019867550,
        "conclusion": "NO_SCALING_SIGNAL",
    },
    "attentive_readout": {
        "classification": "TECHNICAL_NON_RESULT_LINE_CLOSED",
        "scientific_successor_commit": "89dde156d56aaa32d94fae9c54c8eec26b15c8cd",
        "scientific_contract_digest": "e3488a0465d86356fd7e12903cfff7b323ac695b5a50a48ea91df6700dfb5b74",
        "scientific_attempt_digest": "782e504da75ce85e39bebbc4522c375d383ed1a5a5492910affb4969a0c5783e",
        "final_checkpoint_sha256": "f60f0efca0d09df8bdf596948c3cfff0a1bd8dd3913a6cf87d155573d294ce6e",
        "closed_evidence_digest": "bd63b21887694e074b14e9663de47bc8f9f32b84f00e11feed47c9f0a03869c0",
        "terminal_failure_digest": "c6f73db1302f0e53df8cf5d09631646b57fd6c548c7ebb389ecc1da488b729d6",
        "metric_reconstruction_or_publication_authorised": False,
    },
}

TARGET_AVAILABILITY = {
    "horizon_counts": {
        "H1": {"train": 3_922, "selection": 475},
        "H2": {"train": 3_922, "selection": 475},
        "H3": {"train": 3_892, "selection": 471},
        "H4": {"train": 3_854, "selection": 466},
    },
    "horizon_family_counts": {
        "H1": {
            "train": dict(zip(
                FAMILIES, (486, 374, 551, 465, 528, 530, 492, 496),
                strict=True)),
            "selection": dict(zip(
                FAMILIES, (62, 62, 61, 60, 62, 60, 46, 62),
                strict=True)),
        },
        "H2": {
            "train": dict(zip(
                FAMILIES, (486, 374, 551, 465, 528, 530, 492, 496),
                strict=True)),
            "selection": dict(zip(
                FAMILIES, (62, 62, 61, 60, 62, 60, 46, 62),
                strict=True)),
        },
        "H3": {
            "train": dict(zip(
                FAMILIES, (485, 370, 547, 462, 521, 526, 485, 496),
                strict=True)),
            "selection": dict(zip(
                FAMILIES, (62, 62, 60, 60, 61, 59, 46, 61),
                strict=True)),
        },
        "H4": {
            "train": dict(zip(
                FAMILIES, (483, 364, 543, 457, 514, 522, 479, 492),
                strict=True)),
            "selection": dict(zip(
                FAMILIES, (61, 62, 59, 60, 60, 59, 45, 60),
                strict=True)),
        },
    },
    "horizon_family_count_provenance": (
        "independent read-only reduction of the frozen factorial/two-step "
        "identity, frame-metadata, reset and action-block preimage before "
        "contract issue; runtime rederives the H4 intersection and binds these "
        "predeclared H1-H4 counts"
    ),
    "incremental_exclusions": {
        "H2_to_H3": {
            "train": {"reset_boundary": 8, "endpoint_or_end_of_rollout": 22},
            "selection": {"reset_boundary": 0, "endpoint_or_end_of_rollout": 4},
        },
        "H3_to_H4": {
            "train": {"reset_boundary": 12, "endpoint_or_end_of_rollout": 26},
            "selection": {"reset_boundary": 1, "endpoint_or_end_of_rollout": 4},
        },
    },
    "common_rows": 4_320,
    "common_train_rows": 3_854,
    "common_selection_rows": 466,
    "common_train_family_counts": dict(zip(
        FAMILIES, (483, 364, 543, 457, 514, 522, 479, 492), strict=True)),
    "common_selection_family_counts": dict(zip(
        FAMILIES, (61, 62, 59, 60, 60, 59, 45, 60), strict=True)),
    "common_manifest_preimage_digest": "9857af70e482fdde16074fbacb1b9676565a1936d82de0020588162536b4dd39",
    "stable_id_list_digest": "6eed553a7a3a09ef90be5a55e64209991ec8ef405fbe5981eb7356d0872efe49",
    "factorial_position_list_digest": "26e5f1abea18829d42793893a6237d018962255b3598ffa747d4e23c2fb1b07c",
    "partition_pair_order_digest": "77b5beb0dee824342dd708db3c4ba88d38e8708ecf1df346ec08f71a8aca8185",
    "exclusion_digest": "a9e26628bf750800c35d7cef3d43f5ae7efcc18acf2c0e46e1a85daaa3b55b22",
    "exclusion_digest_schema": "independent rich availability-audit records",
    "runner_excluded_stable_compact_digest":
        "7d2dafc31a8563293165d0d867d8c08fcf4488f8c5d0445cb121bf7ffb48a949",
    "runner_excluded_pair_newline_sha256":
        "f5b7fdd2da598ffb5a123a685885ab6dd593e8b2fdd82e7212beb59973a86d1f",
    "runner_exclusion_totals": {
        "reset_or_episode_boundary": 21,
        "endpoint_or_end_of_rollout": 56,
    },
    "included_horizon_frame_indices": "H1_H4_only_not_source_H0",
    "historical_control_train_row_difference": 68,
    "historical_control_train_row_difference_fraction": 68 / 3_922,
    "sample_matched_controls": False,
    "historical_controls_only": True,
    "new_simulator_data_required": False,
}

TARGET_CACHE_CONTRACT = {
    "reuse_existing_H1_H2_train_targets": True,
    "reuse_existing_H1_H4_selection_targets_for_common_rows": True,
    "encode_only_missing_train_H3_H4": True,
    "missing_dense_cache_bytes_each": 6_061_817_856,
    "missing_dense_cache_bytes_total": 12_123_635_712,
    "dense_cache_shape_each": [3_854, 768, 1_024],
    "dense_cache_dtype": "float16",
    "unique_train_frames_requiring_encoder_execution": 5_398,
    "row_horizon_cache_misses": 5_690,
    "output_entries": 7_708,
    "target_encoder_checkpoint":
        "/home/andrewknowles/.cache/vjepa2_1_vitl_dist_vitG_384.pt",
    "target_encoder_checkpoint_sha256":
        "7ea9b7cb4a75d10644a8a8d42cff9e177b10dca8f02173f0eaf2b0bed82838c6",
    "target_encoder_checkpoint_bytes": 5_151_198_524,
    "target_encoder_digest":
        "15ff78a0205ba138a740f12f6eb9bb3f78bce9c5ba8c2849f7e83489a6b2b6a5",
    "preprocessing_digest":
        "8e6aa177b094ea91d27b3c91bcd8f01835b8be5fc51796d145314982ea930fe5",
    "target_encoder_constructor": "vjepa2_1_vit_large_384",
    "target_encoder_repository_commit":
        "204698b45b3712590f06245fbfba32d3be539812",
    "latent_contract": (
        "raw final-block ViT-L dense tokens rounded to float16; consumers "
        "reload float16 as float32 and apply F.layer_norm over the 1024-D "
        "token dimension"),
    "no_intermediate_encoder_layers": True,
    "no_new_simulator_corpus": True,
}

FROZEN_TRAINING_INPUT_FILES = {
    "temporal_rows": {
        "path": str(CACHE_ROOT / "temporal_rows.jsonl"),
        "sha256": "c2014ada5ca3f74e4517d13d4b1b565d982690fd95a0104964b02d30f21444ec",
        "bytes": 10_772_384,
    },
    "factorial_manifest": {
        "path": str(CACHE_ROOT / "proprio_v1/factorial_manifest.json"),
        "sha256": "8bf59020d24e02fdb11948f3732220df839aa1c3bc8612392ce6baab6b8d629c",
        "bytes": 2_305_678,
    },
    "canonical_cache_map": {
        "path": str(CACHE_ROOT / "proprio_v1/canonical_cache_map.json"),
        "sha256": "c9f4c957817bd999d820ee4d8666277270da39aa3099d92be8e1a57bafb4848c",
        "bytes": 2_712_061,
    },
    "proprio_control_rows": {
        "path": str(CACHE_ROOT / "proprio_v1/proprio_rows.jsonl"),
        "sha256": "7b79d12830f12175c591a87982a20e5df7a8d64cfc40e99dd9cee2dc1ae2543e",
        "bytes": 49_284_629,
    },
    "proprio_control_manifest": {
        "path": str(CACHE_ROOT / "proprio_v1/proprio_manifest.json"),
        "sha256": "4d10dfbb91da6960c43b8ff7b204bc85c13690fc91efe3514ccdfab1f041bf79",
        "bytes": 4_532,
    },
    "normalisation": {
        "path": str(CACHE_ROOT / "proprio_v1/proprio_norm_stats.json"),
        "sha256": "9380b4c6d9b59099e43bba9898e1417c273f88075d1ed122401cbb3272e18f94",
        "bytes": 2_329,
    },
    "two_step_rows": {
        "path": str(CACHE_ROOT / "two_step/two_step_rows.jsonl"),
        "sha256": "42bbcf0cea09426c06e884219dd6bc7df7992313a4f50c0ea6ad8a3a5c6e368d",
        "bytes": 12_833_756,
    },
    "paired_navigation_manifest": {
        "path": str(ROOT / ".generated/go2_paired_navigation/geometry_v3_physical_v1/"
                    "dataset/dataset_manifest.json"),
        "sha256": "ed927cceaedb56ff68334af5109381466740850554048127bb72f04da59f7180",
        "bytes": 403_289,
    },
    "train_context_Hminus2": {
        "path": str(CACHE_ROOT / "temporal_action_jepa_v1/"
                    "predicted_token_diagnostic/frozen_train_ctx0.f16"),
        "sha256": "de85f14150d7aec653cd3d673123854022785e9574a08f11b2a2ceb65db3771e",
        "bytes": 6_409_420_800,
    },
    "train_context_Hminus1": {
        "path": str(CACHE_ROOT / "temporal_action_jepa_v1/"
                    "predicted_token_diagnostic/frozen_train_ctx1.f16"),
        "sha256": "77dc70368d3927e47c28c614b28f97b1a9ba2484e1d9cd3ce5be760a293fc788",
        "bytes": 6_409_420_800,
    },
    "selection_context_Hminus2": {
        "path": str(CACHE_ROOT / "temporal_action_jepa_v1/evaluation/"
                    "frozen_ctx0.f16"),
        "sha256": "1c26e2a83454aee2861a5a7828895c535c718c1d38bd9daca4e2ce62fbe181b1",
        "bytes": 772_276_224,
    },
    "selection_context_Hminus1": {
        "path": str(CACHE_ROOT / "temporal_action_jepa_v1/evaluation/"
                    "frozen_ctx1.f16"),
        "sha256": "1ec5b19a44b4c6f0e950b70945b08f3dcdd8bb15a45a5abfbe3e7f3f61624cec",
        "bytes": 772_276_224,
    },
    "current_context_train_and_selection": {
        "path": str(CACHE_ROOT / "temporal_action_jepa_v1/evaluation/"
                    "frozen_current.f16"),
        "sha256": "3c63e397002bc44c637d4c94e63823c335e61704ecee9cdf9b4950e5df8b6abc",
        "bytes": 7_181_697_024,
    },
    "train_target_H1": {
        "path": str(CACHE_ROOT / "temporal_action_jepa_v1/evaluation/"
                    "frozen_train_future.f16"),
        "sha256": "e973dd046e5868f190809a0ff5b5b2b7714652cf6c66b2e62abe2c72b884d8d4",
        "bytes": 6_409_420_800,
    },
    "selection_target_H1": {
        "path": str(CACHE_ROOT / "temporal_action_jepa_v1/evaluation/"
                    "frozen_sel_future.f16"),
        "sha256": "d4b3560dac4ea1d3d9149dd137ce514b457de1ece5b9ada746a0ceb9650b364c",
        "bytes": 772_276_224,
    },
    "train_target_H2": {
        "path": str(CACHE_ROOT / "two_step/frozen_train_step2.f16"),
        "sha256": "6b68e048522ea225236d3d7161422f373bf7cd4876b61401fafede5cd54f55f1",
        "bytes": 6_340_214_784,
    },
    "selection_target_H2": {
        "path": str(CACHE_ROOT / "two_step/frozen_sel_step2.f16"),
        "sha256": "9cf4cd2fa6b0e4b1a83f86688ec05283b6c17ed4267e3f00ab0b4c7c85ac0a96",
        "bytes": 767_557_632,
    },
    "selection_target_H3": {
        "path": str(CACHE_ROOT / "horizons/target_h3.f16"),
        "sha256": "eec29b4843145f3c3d981fa6d97b3f46bde402ccfbefd9e0be2c8c0de06addbb",
        "bytes": 753_401_856,
    },
    "selection_horizon_rows": {
        "path": str(CACHE_ROOT / "horizons/FINAL/FINAL_horizon_rows_479.jsonl"),
        "sha256": "644a257803b5d49dc05a8e5b90b057b1558e2b4c22208f64070d2cc218fce0cd",
        "bytes": 1_990_279,
    },
    "selection_target_H4": {
        "path": str(CACHE_ROOT / "horizons/target_h4.f16"),
        "sha256": "f357cbc0440c16d47a1bf4143c8b2f5a5a9f5b917e5af91f7b95807f24a55d24",
        "bytes": 753_401_856,
    },
}

MODEL_AND_OBJECTIVE = {
    "model": "scripts.dev_proprio_predictor_v1.ProprioActionPredictor",
    "use_proprio": False,
    "input_cell": "RGB_PLUS_CONTROL_HISTORY",
    "token_shape": [768, 1_024],
    "predictor_width": 384,
    "predictor_depth": 6,
    "attention_heads": 6,
    "context_slots": 3,
    "control_history_shape": [3, 5, 2],
    "action_block_shape": [5, 2],
    "action_dimensions": 10,
    "autoregressive_horizons": [1, 2, 3, 4],
    "own_preceding_prediction_after_H1": True,
    "teacher_forcing_after_H1": False,
    "future_proprioception": False,
    "detach_preceding_prediction": False,
    "per_horizon_loss": (
        "mean absolute error over batch, 768 tokens and 1024 dimensions on "
        "the frozen normalized latent representation"),
    "per_horizon_loss_function": "torch.nn.functional.l1_loss/default mean",
    "aggregate_loss": "(L1 + L2 + L3 + L4) / 4",
    "aggregate_loss_scale": 0.25,
    "horizon_specific_weights": False,
    "architecture_change": False,
    "utility_occupancy_safety_or_proprio_loss": False,
    "target_encoder_stop_gradient_unchanged": True,
    "prediction_normalisation": (
        "scripts.dev_proprio_predictor_v1.unroll applies frozen token-axis "
        "normalisation after every predicted step"),
}

TRAINING = {
    "seeds": list(FROZEN_SEEDS),
    "data_order_seed_by_run": {seed: seed for seed in FROZEN_SEEDS},
    "base_initial_weights": "same seed-specific registered base weights as historical pair",
    "data_order": "filtered historical 3922-row batch plan; see data_order_contract",
    "augmentation": "none; historical plan preserved",
    "epochs": 24,
    "checkpoint_epoch": 21,
    "checkpoint_selection": False,
    "batch_size": 4,
    "optimizer": "AdamW",
    "learning_rate": 3e-4,
    "weight_decay": 1e-2,
    "optimizer_foreach": False,
    "schedule": "fixed learning rate; no scheduler",
    "gradient_clip": 1.0,
    "precision": "bf16 autocast with frozen FP32 optimizer/model-state contract",
    "dropout": "disabled",
    "terminal_window_epochs": [19, 20, 21, 22, 23],
    "terminal_window_used_for_selection_or_exclusion": False,
    "finite_weak_runs_retained": True,
    "extension_or_best_epoch": False,
    "run_count": 8,
}

DATA_ORDER_CONTRACT = {
    "source_function": "scripts.run_dev_proprio_factorial_driver_v1.batch_plan",
    "historical_plan_arguments": "batch_plan(seed, epoch, 3922, 4)",
    "historical_train_rows": 3_922,
    "common_train_rows": 3_854,
    "excluded_historical_train_rows": 68,
    "algorithm": (
        "for each seed and epoch, flatten the frozen historical 3922-row "
        "batch_plan in its emitted order; remove only original factorial train "
        "indices absent from the H4-valid common manifest; remap surviving "
        "indices to common-manifest train positions; then chunk consecutively "
        "into batch size four, retaining the final partial batch"),
    "direct_batch_plan_on_3854_forbidden": True,
    "additional_rng_draws": 0,
    "survivor_relative_order_identical_to_historical": True,
    "per_seed_epoch_order_digests": (
        "freeze all 8 x 24 sequence digests in the common-manifest receipt "
        "before smoke or scientific training"),
    "per_seed_epoch_batch_digests": (
        "freeze all 8 x 24 rechunked-batch digests in the common-manifest "
        "receipt before smoke or scientific training"),
    "historical_comparator_sample_mismatch_remains": True,
}

SMOKE_GATES = {
    "one_tiny_real_feature_forward_backward": True,
    "objective_separation_H1_H4": True,
    "each_component_perturbation_changes_only_its_registered_Li": True,
    "combined_loss_derivative_per_component": 0.25,
    "all_parameter_gradients_finite": True,
    "H3_H4_backpropagate_through_autoregressive_chain": True,
    "adaln_zero_warmup_permitted_only_for_chain_test": True,
    "chain_check_warmup_steps": 50,
    "warmup_state_discarded": True,
    "exact_registered_base_reloaded_after_smoke": True,
    "checkpoint_save_resume_exact": True,
    "broad_implementation_audit": False,
}

RESOURCE_GATES = {
    "preflight_full_epochs": 1,
    "preflight_seed": FROZEN_SEEDS[0],
    "preflight_weights_discarded_not_a_scientific_run": True,
    "registered_base_reloaded_for_seed_run": True,
    "peak_vram_strictly_below_bytes": 28 * 2**30,
    "free_system_ram_strictly_above_bytes": 20 * 2**30,
    "filesystem_capacity_must_cover": [
        "H3_H4_target_caches", "eight_epoch21_checkpoints",
        "eight_training_receipts", "predictions", "metric_receipts"],
    "batch_size_change_after_launch": False,
    "measure_wall_seconds_per_epoch": True,
    "measure_peak_vram": True,
    "measure_peak_system_ram": True,
    "project_eight_run_runtime": True,
    "measure_target_cache_and_checkpoint_storage": True,
}

FROZEN_EVALUATION = {
    "stage_root": str(ROOT / ".generated/go2_counterfactual_fidelity_v1_2"),
    "predictor_result_path": str(
        ROOT / ".generated/go2_counterfactual_fidelity_v1_2/"
        "predictor_assay/result.json"),
    "stage_a_identity_manifest_digest":
        "ce2cbbe8dab9a89ad6f85d16c56a9d712d791c8bbfd8925a8f01efc0c039705a",
    "corpus_digest":
        "f84eb3271f1a3b7052bbf2e84240453e84772b0a530e60ec47f723a44e2e10e9",
    "branch_rows_sha256":
        "2b71c488851c6d4b7e3a36a46637a4e5be4896ae48a84d1498c6e8a8d3d74c81",
    "completion_receipt_digest":
        "b448775b8c62539e5b5f9b3c1f0d2d86da85f40311a38a4f6b2cef550cbb0c2f",
    "latent_index_digest":
        "861285ec9c8fc6c92c6f3a31cade0f031172bf6818d76d1899634a60c7e5c291",
    "verified_latent_shard_set_digest":
        "eeb381e28d851db60d4341654860b77e9a0aef0abae1c3e7673d75d82bc5916f",
    "assay_spec_digest":
        "a26fa0ec9ee9e0df3bbe71fff6d7594bb714227aaa66a66631836d94a676feab",
    "frozen_predictor_result_digest":
        "3b5c500b4b1326056ce18c6276d7842f4230faec36f8f29cc65945f54527bbcb",
    "frozen_predictor_result_file_sha256":
        "d3f5ade362a2df4546d3c6cfe7d5f3fc1d3ee0216fa13eef4cba0e2a48f028be",
    "frozen_predictor_result_bytes": 15_199_537,
    "states": 20,
    "branches": 240,
    "candidates_per_state": 12,
    "families": 8,
    "horizons": [1, 2, 3, 4],
    "regenerate_any_input_or_label": False,
    "regenerate_branches_target_latents_labels_or_occupancy_masks": False,
    "historical_comparator_model_forward": False,
    "new_four_step_model_forward_only": True,
}

FROZEN_METRIC_DEFINITIONS = {
    "changed_token_mask": {
        "frozen_source": str(
            CACHE_ROOT / "two_step/evaluation/"
            "MATCHED_24_EPOCH_result_epochs_0_23.json"),
        "frozen_source_sha256":
            "65cac7353b8542fb3a35354864e2da79d356cf6253c2876ccf64d647fc69c71d",
        "H1_threshold": 0.7618998289108276,
        "H2_H4_threshold": 0.8970220685005188,
        "assay_data_threshold_fitting": False,
    },
    "changed_token_correct_future_cosine": (
        "mean token cosine(prediction,target) over the target-specific frozen "
        "changed-token mask"),
    "persistence_changed_cosine": (
        "mean token cosine(last observed context,target) over the same mask"),
    "advantage_over_persistence": (
        "changed_token_correct_future_cosine minus persistence_changed_cosine"),
    "normalized_error": (
        "mean_changed_tokens(mean_dim((prediction-target)^2)) divided by "
        "max(mean_changed_tokens(mean_dim((last_context-target)^2)),1e-12)"),
    "retrieval": {
        "queries": "each predicted candidate trajectory",
        "gallery": "the twelve registered true branches in the same state",
        "similarity": (
            "mean token cosine(pred_i,target_j) over the complete aligned "
            "768-token grid"),
        "changed_masks_used": False,
        "correct_identity": "candidate i equals target candidate j",
        "tie_rule": "descending cosine then frozen target candidate index",
        "tie_atol": 1e-12,
        "metrics": [
            "top1", "top3", "mean_reciprocal_rank", "mean_rank",
            "median_rank", "mean_margin_over_best_wrong",
            "mean_margin_over_mean_wrong", "pairwise_accuracy", "confusion"],
    },
    "direct_aggregation": (
        "candidate row mean, then state/episode-cluster mean, then family "
        "mean, then unweighted mean of all eight frozen families"),
    "corpus_weighted": "reported separately from the equal-family primary",
    "normalisation": TARGET_CACHE_CONTRACT["latent_contract"],
}

PRIMARY_ENDPOINTS_H4 = (
    "changed_token_correct_future_cosine",
    "normalized_error_reduction",
    "correct_branch_top1_retrieval",
    "mean_reciprocal_rank",
    "pairwise_branch_discrimination",
)
SECONDARY_ENDPOINTS = (
    "H1_H3_direct_fidelity", "H1_H3_retrieval",
    "H1_H4_degradation_slopes", "persistence_baseline_comparison",
    "own_vs_best_other_margin", "own_vs_mean_other_margin",
    "candidate_confusion_patterns", "terminal_window_stability",
)
STATISTICAL_CONTRACT = {
    "primary_effect": "four_step minus two_step, oriented so positive is better",
    "normalized_error_effect": "two_step error minus four_step error",
    "replication_unit": "eight paired training seeds",
    "paired_effect_count": 8,
    "interval": "two-sided 95% Student t interval using sample SD and df=7",
    "sample_standard_deviation_ddof": 1,
    "degrees_of_freedom": 7,
    "two_sided_95_percent_t_critical": 2.3646242510102993,
    "paired_summary_fields": [
        "eight_seed_effects", "mean", "sample_sd", "ci95_low", "ci95_high"],
    "primary_weighting": "equal-family",
    "separate_weighting": "corpus-weighted",
    "report_per_family": True,
    "report_one_two_four_step_cell_means": True,
    "ties_and_retrieval": "frozen predictor-assay definitions",
}

OCCUPANCY = {
    "result_path": str(
        ROOT / ".generated/go2_counterfactual_fidelity_v1_2/"
        "occupancy_results/result.json"),
    "probe_package_digest":
        "b8f05e57baffcf553ba9581419d82068a5723f2aae5895de29b9546d4c3f7686",
    "probe_package_file_sha256":
        "3d216f4e60851861d521705397ae0f43f783a8ceb1852685f42ab27ff0260c75",
    "probe_specification_digest":
        "646073a9b0a43d7a6c3230f55b3d68026d0632af70726c196603cb7ccf182478",
    "probe_weights_sha256":
        "95d253ce834384f1b372f1c4cc7f39241c42576fdea903c007dda8f7a7bc1322",
    "probe_weights_bytes": 100_785_421,
    "probe_state_digest":
        "588295858ab326f31084e542bd1d86c23b5d08defe41567533e3b12bd10c84ac",
    "frozen_assay_spec_digest":
        "336c796d6256934492edf67650ddd0b71f3c661a5c9610b89ad8abff9c51fca1",
    "frozen_result_digest":
        "09dc413d9ce30c2cb19c99e93eeaad410983a7f53575387bc6694f3844a070d6",
    "frozen_result_file_sha256":
        "f9e6e47f8b8208e00b31836b5347424c368b7a5dcf96d9037cf9925e04d1a0af",
    "frozen_result_bytes": 342_950,
    "true_target_gate_digest":
        "4bf9a92144fa728d953c9dffebb235c9b476ded59d7462a107fe2e6ade0894e4",
    "true_target_observable_occupied_IoU_floor": 0.35,
    "qualified_true_target_horizons": [2, 3, 4],
    "row_metric": "observable occupied IoU; undefined union is NaN",
    "primary_aggregation": (
        "row within episode cluster, cluster within family, unweighted mean "
        "over eight families"),
    "secondary_aggregation": "unweighted mean over defined branch rows",
    "H1_unavailable_and_not_reinterpreted": True,
    "refit": False,
    "co_outcome_not_formal_non_regression": True,
}

INTERPRETATION = {
    "H4_direct_fidelity_endpoints": [
        "changed_token_correct_future_cosine",
        "normalized_error_reduction",
    ],
    "useful_requires_both_H4_direct_fidelity_paired_means_strictly_positive": True,
    "useful_requires_improved_H4_top1_or_pairwise": True,
    "retrieval_improvement_rule": (
        "the equal-family H4 paired mean is strictly positive for correct-branch "
        "top-1 or pairwise branch discrimination; no CI-exclusion rule is added"),
    "material_H1_H2_regression_rule": (
        "material early-horizon regression is present if the equal-family paired "
        "95% t interval lies wholly below zero for either changed-token correct-"
        "future cosine benefit or normalized-error-reduction benefit at H1 or H2"),
    "discordant_H4_direct_signs": (
        "DIRECT_FIDELITY_EVIDENCE_DISCORDANT_OR_MIXED; useful=false; report both "
        "effects and intervals without post-hoc endpoint selection"),
    "H4_improves_with_material_H1_H2_regression": "REPORT_HORIZON_TRADEOFF",
    "direct_improves_retrieval_does_not": (
        "LONGER_ROLLOUT_IMPROVES_LATENT_ACCURACY_WITHOUT_CANDIDATE_DIFFERENTIATION"),
    "H4_improves_H1_H2_regress": "REPORT_HORIZON_TRADEOFF",
    "planning_improvement_claim": False,
    "utility_or_selected_action_endpoint": False,
}

STAGES = (
    "verify_frozen_comparators",
    "audit_and_freeze_common_H4_manifest",
    "encode_missing_H3_H4_train_targets",
    "tiny_real_feature_smoke",
    "one_full_epoch_resource_preflight",
    "train_exactly_eight_four_step_runs",
    "evaluate_frozen_240_branch_corpus",
    "score_frozen_occupancy_probe_H2_H4",
    "validate_and_terminalize",
)
RUNNER_STAGES = (
    "issue", "manifest", "encode", "smoke", "preflight",
    "train-seed", "train-all", "evaluate", "validate",
)
OUTPUT_PATHS = {
    "contract": "contract.json",
    "target_audit": "target_availability.json",
    "common_manifest": "common_h4_manifest.json",
    "common_rows": "common_h4_rows.jsonl",
    "target_index": "target_cache_index.json",
    "smoke": "smoke.json",
    "preflight": "resource_preflight.json",
    "training_receipts": "training_receipts.json",
    "prediction_ledgers": "evaluation/prediction_ledgers",
    "evaluation": "evaluation/result.json",
    "occupancy": "evaluation/occupancy.json",
    "terminal": "terminal.json",
}

AUTHORITY = {
    "train_eight_four_step_RGB_control_history_models": True,
    "evaluate_only_epoch21_four_step_checkpoints": True,
    "reuse_historical_RGB_one_step_and_two_step_metrics": True,
    "retrain_or_reselect_historical_comparators": False,
    "use_proprioceptive_cells": False,
    "new_simulator_corpus_or_branch_generation": False,
    "utility_scorer_or_predictor_utility_shard_access": False,
    "selected_action_or_planning_endpoint": False,
    "sealed_or_held_out_access": False,
    "attentive_readout_metric_reconstruction_or_execution": False,
    "train_another_utility_readout_or_try_ViT_G": False,
    "modify_any_existing_scientific_result": False,
    "longer_H5_plus_hierarchical_or_architecture_variant": False,
    "navigation_or_final_corpus": False,
    "final_200_state_utility_ranking_corpus": False,
    "retry_extend_reseed_or_best_epoch_select": False,
}

ENVIRONMENT_REFERENCE = {
    "interpreter": "/home/andrewknowles/TinyQuadJEPA/bin/python",
    "torch_module": (
        "/home/andrewknowles/TinyQuadJEPA/lib/python3.12/site-packages/"
        "torch/__init__.py"),
    "historical_python": "3.12.3",
    "historical_torch": "2.10.0.dev20250926+rocm6.3",
    "historical_hip": "6.3.42131-fa1d09cbd",
    "historical_device": "AMD Radeon AI PRO R9700",
    "historical_device_memory_gib": 31.9,
    "exact_environment_required_for_smoke_preflight_train_and_evaluate": True,
    "runtime_environment_must_be_recorded": True,
    "environment_identity_claim": True,
}

TERMINAL_KINDS = (
    "COMPLETE_FOUR_STEP_ROLLOUT_OBJECTIVE_RESULT",
    "INVALID_CONTRACT_OR_SOURCE_LINEAGE",
    "INVALID_FROZEN_COMPARATOR_LINEAGE",
    "INVALID_TARGET_AVAILABILITY",
    "INVALID_TARGET_ENCODING_OR_CACHE",
    "INVALID_SMOKE",
    "INVALID_RESOURCE_PREFLIGHT",
    "INVALID_TRAINING",
    "INVALID_EVALUATION",
    "INVALID_OCCUPANCY_CO_OUTCOME",
)
FAILURE_STAGE_CLASSIFICATION = {
    "issue": "INVALID_CONTRACT_OR_SOURCE_LINEAGE",
    "manifest_lineage": "INVALID_FROZEN_COMPARATOR_LINEAGE",
    "manifest_availability": "INVALID_TARGET_AVAILABILITY",
    "encode": "INVALID_TARGET_ENCODING_OR_CACHE",
    "smoke": "INVALID_SMOKE",
    "preflight": "INVALID_RESOURCE_PREFLIGHT",
    "train-seed": "INVALID_TRAINING",
    "train-all": "INVALID_TRAINING",
    "evaluate": "INVALID_EVALUATION",
    "occupancy": "INVALID_OCCUPANCY_CO_OUTCOME",
    "validate": "INVALID_EVALUATION",
}
FAILURE_RECEIPT_REQUIRED_FIELDS = (
    "classification", "failed_stage", "exception_type", "exception",
    "contract_digest", "source_commit", "common_manifest_digest_if_issued",
    "targets_encoded", "completed_training_seed_count",
    "completed_training_seeds", "completed_evaluation_seed_count",
    "artifacts_present", "retry_resume_or_replacement_authorised",
    "nothing_remains_running",
)
DELIVERABLES = (
    "training_target_availability",
    "common_manifest_digest",
    "four_step_contract_digest",
    "smoke_result",
    "resource_preflight",
    "eight_training_receipts",
    "complete_H1_H4_direct_fidelity",
    "complete_H1_H4_retrieval",
    "paired_four_step_vs_two_step_analysis",
    "occupancy_co_outcomes_H2_H4",
    "runtime_and_storage",
    "technical_invalidity_if_any",
    "confirmation_no_utility_planning_or_new_corpus_access",
    "confirmation_nothing_remains_running",
)
TERMINAL_REQUIREMENTS = {
    "all_started_child_processes_joined": True,
    "training_or_evaluation_processes_remaining": 0,
    "target_encoder_processes_remaining": 0,
    "no_automatic_follow_on_experiment": True,
    "no_predictor_utility_scoring": True,
    "no_final_corpus_generation": True,
}

HEX64 = re.compile(r"[0-9a-f]{64}")


class FourStepContractError(RuntimeError):
    """The frozen source, lineage, data, or one-shot contract changed."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise FourStepContractError(message)


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=True, allow_nan=False).encode("ascii")


def digest(value: Any) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def file_sha256(path: Path, block_size: int = 8 << 20) -> str:
    result = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(block_size), b""):
            result.update(block)
    return result.hexdigest()


def _git(root: Path, *arguments: str) -> str:
    try:
        return subprocess.check_output(
            ["git", *arguments], cwd=root, text=True,
            stderr=subprocess.STDOUT).strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise FourStepContractError(f"cannot bind four-step source: {exc}") from exc


def source_closure(root: Path = ROOT) -> dict[str, Any]:
    require(_git(root, "status", "--porcelain=v1") == "",
            "four-step source must be clean and committed")
    head = _git(root, "rev-parse", "HEAD")
    require(subprocess.run(
        ["git", "merge-base", "--is-ancestor", BASE_SOURCE_COMMIT, head],
        cwd=root, stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL).returncode == 0,
        "four-step source does not descend from frozen base")
    changed = tuple(sorted(filter(None, _git(
        root, "diff", "--name-only", f"{BASE_SOURCE_COMMIT}..{head}"
    ).splitlines())))
    require(changed == tuple(sorted(NEW_SOURCE_PATHS)),
            "committed diff is not exactly the four additive four-step paths")
    frozen: dict[str, dict[str, Any]] = {}
    for relative, (sha256, byte_count) in FROZEN_SOURCE_FILES.items():
        target = root / relative
        require(target.is_file() and not target.is_symlink()
                and target.stat().st_size == byte_count
                and file_sha256(target) == sha256,
                f"frozen predictive source changed: {relative}")
        frozen[relative] = {"sha256": sha256, "byte_count": byte_count}
    additive: dict[str, dict[str, Any]] = {}
    for relative in NEW_SOURCE_PATHS:
        target = root / relative
        require(target.is_file() and not target.is_symlink(),
                f"additive four-step source absent: {relative}")
        additive[relative] = {
            "sha256": file_sha256(target),
            "byte_count": target.stat().st_size,
        }
    payload = {
        "schema": SOURCE_SCHEMA,
        "base_source_commit": BASE_SOURCE_COMMIT,
        "source_repository_commit": head,
        "source_repository_clean": True,
        "exact_committed_additive_path_diff": list(changed),
        "frozen_source_files": frozen,
        "additive_files": additive,
    }
    return {**payload, SOURCE_SELF_KEY: digest(payload)}


def runtime_root(root: Path = ROOT) -> Path:
    del root
    return REGISTERED_RUNTIME_PARENT / RUNTIME_RELATIVE


def contract_path(root: Path = ROOT) -> Path:
    return runtime_root(root) / "contract.json"


def storage_binding(root: Path = ROOT) -> dict[str, Any]:
    target = runtime_root(root)
    require(not target.exists() and not target.is_symlink(),
            "one-shot four-step runtime namespace already exists")
    return {
        "registered_runtime_parent": str(REGISTERED_RUNTIME_PARENT),
        "runtime_relative": str(RUNTIME_RELATIVE),
        "runtime_path": str(target),
        "runtime_namespace_absent_before_issue": True,
        "workspace_generated_output": False,
    }


def static_contract() -> dict[str, Any]:
    return {
        "status": STATUS,
        "label": LABEL,
        "stages": list(STAGES),
        "runner_stages": list(RUNNER_STAGES),
        "output_paths": OUTPUT_PATHS,
        "frozen_seeds": list(FROZEN_SEEDS),
        "families": list(FAMILIES),
        "base_weight_sha256": BASE_WEIGHT_SHA256,
        "base_state_digest": BASE_STATE_DIGEST,
        "base_weight_bytes": BASE_WEIGHT_BYTES,
        "comparator_checkpoint_bytes": COMPARATOR_CHECKPOINT_BYTES,
        "comparator_checkpoint_sha256": COMPARATOR_CHECKPOINT_SHA256,
        "comparator_objectives": COMPARATOR_OBJECTIVES,
        "frozen_factorial": FROZEN_FACTORIAL,
        "closed_scientific_lines": CLOSED_SCIENTIFIC_LINES,
        "target_availability": TARGET_AVAILABILITY,
        "target_cache_contract": TARGET_CACHE_CONTRACT,
        "frozen_training_input_files": FROZEN_TRAINING_INPUT_FILES,
        "model_and_objective": MODEL_AND_OBJECTIVE,
        "training": TRAINING,
        "data_order_contract": DATA_ORDER_CONTRACT,
        "smoke_gates": SMOKE_GATES,
        "resource_gates": RESOURCE_GATES,
        "frozen_evaluation": FROZEN_EVALUATION,
        "frozen_metric_definitions": FROZEN_METRIC_DEFINITIONS,
        "primary_endpoints_H4": list(PRIMARY_ENDPOINTS_H4),
        "secondary_endpoints": list(SECONDARY_ENDPOINTS),
        "statistical_contract": STATISTICAL_CONTRACT,
        "occupancy": OCCUPANCY,
        "interpretation": INTERPRETATION,
        "environment_reference": ENVIRONMENT_REFERENCE,
        "terminal_kinds": list(TERMINAL_KINDS),
        "failure_stage_classification": FAILURE_STAGE_CLASSIFICATION,
        "failure_receipt_required_fields": list(FAILURE_RECEIPT_REQUIRED_FIELDS),
        "deliverables": list(DELIVERABLES),
        "terminal_requirements": TERMINAL_REQUIREMENTS,
        "authority": AUTHORITY,
    }


def build_contract(source: Mapping[str, Any], storage: Mapping[str, Any]) -> dict[str, Any]:
    require(source.get("schema") == SOURCE_SCHEMA
            and source.get(SOURCE_SELF_KEY) == digest({
                key: value for key, value in source.items()
                if key != SOURCE_SELF_KEY}),
            "four-step source closure changed")
    require(storage.get("runtime_namespace_absent_before_issue") is True
            and storage.get("runtime_path") == str(runtime_root()),
            "four-step storage binding changed")
    payload = {
        "schema": CONTRACT_SCHEMA,
        "source_closure": dict(source),
        "storage": dict(storage),
        **static_contract(),
    }
    return {**payload, CONTRACT_SELF_KEY: digest(payload)}


def validate_contract(value: Mapping[str, Any]) -> dict[str, Any]:
    result = dict(value)
    recorded = result.pop(CONTRACT_SELF_KEY, None)
    require(isinstance(recorded, str) and HEX64.fullmatch(recorded) is not None
            and recorded == digest(result), "four-step contract self digest changed")
    result[CONTRACT_SELF_KEY] = recorded
    require(canonical_bytes(result) == canonical_bytes(build_contract(
                result["source_closure"], result["storage"])),
            "four-step contract changed")
    return result


def validate_installed_source(value: Mapping[str, Any],
                              root: Path = ROOT) -> dict[str, Any]:
    """Validate a receipt and re-bind the live clean installed source bytes."""
    result = validate_contract(value)
    observed = source_closure(root)
    require(canonical_bytes(observed)
            == canonical_bytes(result.get("source_closure")),
            "installed four-step source differs from issued source closure")
    return result


__all__ = [name for name in globals() if name.isupper()] + [
    "FourStepContractError", "build_contract", "canonical_bytes",
    "contract_path", "digest", "file_sha256", "require", "runtime_root",
    "source_closure", "static_contract", "storage_binding", "validate_contract",
    "validate_installed_source",
]
