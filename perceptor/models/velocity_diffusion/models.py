from . import cc12m_1, yfcc_1, yfcc_2, wikiart_256


MODELS = {
    "cc12m_1": cc12m_1.CC12M1Model,
    "cc12m_1_cfg": cc12m_1.CC12M1Model,
    "yfcc_1": yfcc_1.YFCC1Model,
    "yfcc_2": yfcc_2.YFCC2Model,
    "wikiart": wikiart_256.WikiArt256Model,
}


def get_model(model):
    return MODELS[model]


def get_models():
    return list(MODELS.keys())
