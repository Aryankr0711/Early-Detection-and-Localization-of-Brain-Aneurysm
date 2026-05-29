import unittest

from ui2 import server

try:
    import torch
    from src.models.aneurysm_vessel_seg_roi_module import AneurysmVesselSegROILitModule
except ModuleNotFoundError:
    torch = None
    AneurysmVesselSegROILitModule = None


class BackendResultTests(unittest.TestCase):
    def test_fallback_probability_row_is_detected(self):
        row = {
            label: value
            for label, value in zip(
                server.ANEURYSM_CLASSES,
                [0.02, 0.02, 0.08, 0.08, 0.03, 0.03, 0.07, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.35],
            )
        }

        self.assertTrue(server.is_error_fallback_row(row))

    def test_non_fallback_probability_row_is_not_detected(self):
        row = {label: 0.01 for label in server.ANEURYSM_CLASSES}
        row["Aneurysm Present"] = 0.91

        self.assertFalse(server.is_error_fallback_row(row))

    def test_legacy_response_includes_patient_metadata(self):
        result = {
            "source": "real",
            "aneurysm_present_probability": 0.12,
            "probabilities": [
                {"label": "L-IC Infra", "value": 0.05},
                {"label": "Aneurysm Present", "value": 0.12},
            ],
            "metadata": {
                "patient_id": "1.2.3",
                "age": "64",
                "sex": "F",
                "modality": "MRA",
            },
            "notes": [],
        }

        legacy = server.to_legacy_analysis_response(result)

        self.assertEqual(legacy["patientId"], "1.2.3")
        self.assertEqual(legacy["patientAge"], "64 / F")
        self.assertEqual(legacy["patientModality"], "MRA")

    def test_roi_forward_guard_casts_cpu_half_inputs_to_float(self):
        if torch is None or AneurysmVesselSegROILitModule is None:
            self.skipTest("torch is not available in this test runtime")
        module = object.__new__(AneurysmVesselSegROILitModule)
        seen = {}

        class Runtime:
            def get_runtime_module(self, use_ema=None):
                return self

            def _forward_impl(self, x, vessel_seg=None, vessel_union=None):
                seen["x_dtype"] = x.dtype
                seen["vessel_dtype"] = vessel_seg.dtype
                seen["union_dtype"] = vessel_union.dtype
                raise RuntimeError("stop after dtype capture")

        module.model = Runtime()

        with self.assertRaisesRegex(RuntimeError, "stop after dtype capture"):
            AneurysmVesselSegROILitModule.forward(
                module,
                torch.zeros((1, 1, 2, 2, 2), dtype=torch.float16),
                torch.zeros((1, 13, 2, 2, 2), dtype=torch.float16),
                torch.zeros((1, 1, 2, 2, 2), dtype=torch.float16),
            )

        self.assertEqual(seen["x_dtype"], torch.float32)
        self.assertEqual(seen["vessel_dtype"], torch.float32)
        self.assertEqual(seen["union_dtype"], torch.float32)


if __name__ == "__main__":
    unittest.main()
