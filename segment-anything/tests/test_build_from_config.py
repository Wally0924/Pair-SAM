# segment-anything/tests/test_build_from_config.py
"""
執行：conda run -n sam_env python -m pytest segment-anything/tests/test_build_from_config.py -v
不載入 SAM checkpoint（checkpoint=None），僅驗證 config→屬性映射。vit_b 約 4s/次。
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from segment_anything.build_weather_sam import build_weather_sam_from_config


def _cfg(**over):
    base = dict(model_type='vit_b', use_vgg_adapter=True, inject='pre',
                decoder='unified', lrh=True, mfb=True, ref=True)
    base.update(over)
    return base


def test_config_maps_to_attributes():
    m = build_weather_sam_from_config(_cfg(decoder='per_class', lrh=False, ref=False),
                                      checkpoint=None)
    assert m.mask_decoder.decoder_mode == 'per_class'
    assert m.use_lrh is False
    assert m.vgg_injector.use_reference is False


def test_full_defaults_backward_compatible():
    m = build_weather_sam_from_config(_cfg(), checkpoint=None)
    assert m.mask_decoder.decoder_mode == 'unified'
    assert m.use_lrh is True
    assert m.vgg_injector.use_reference is True
