"""A component designator must never masquerade as its library model."""
import unittest
from unittest.mock import Mock, patch

from stage2 import assemble, libpins


class ModelLibraryIdentityTests(unittest.TestCase):
    def build(self, model=None, name="OBSERVED"):
        stage = dict(image=dict(w=300, h=300),
                     components=[dict(id="ic", type="box", bbox_img=[100, 100, 200, 200],
                                      bbox_json=[100, 100, 200, 200])],
                     pins={"ic": [[212, 150]]}, nets=[])
        rec = dict(designator=dict(text="U1"), pins={0: dict(
            number=dict(text="1", conf=.98), name=None if name is None else dict(text=name, conf=.98))})
        if model is not None:
            rec["partname"] = dict(text=model, conf=.99)
        ctx = Mock()
        ctx.key_for.return_value = "MODEL"
        ctx.names_of.return_value = {}
        ctx.name_for_num.return_value = "LIBRARY_NAME"
        with patch.object(assemble, "LIB_PINS", True), \
                patch.object(assemble, "BOX_PIN_NEED_NAME", True), \
                patch.object(assemble, "BOX_PIN_NEED_NAME_LATE", True), \
                patch.object(libpins, "new_ctx", return_value=ctx), \
                patch.object(libpins, "degen_mode", return_value="off"):
            doc = assemble.build_doc(stage, dict(comps={"ic": rec}), "library-identity")
        return doc, ctx

    def test_missing_model_preserves_observed_name_without_lookup(self):
        doc, ctx = self.build()
        self.assertEqual(doc["pins"]["U1"]["pin_1"]["pinname"], "OBSERVED")
        ctx.key_for.assert_not_called()

    def test_designator_repeated_as_model_cannot_enable_lookup(self):
        doc, ctx = self.build(model="U1", name=None)
        self.assertEqual(doc["pins"]["U1"], {})
        ctx.key_for.assert_not_called()

    def test_identified_model_retains_upstream_library_completion(self):
        doc, ctx = self.build(model="PART123", name=None)
        self.assertEqual(doc["pins"]["U1"]["pin_1"]["pinname"], "LIBRARY_NAME")
        ctx.key_for.assert_called_with("PART123")


if __name__ == "__main__":
    unittest.main()
