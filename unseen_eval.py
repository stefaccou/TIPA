import re
from collections import OrderedDict
import torch
import evaluate
from huggingface_hub import HfApi
from qq import LanguageData, TagType
from urielplus import urielplus


metric = evaluate.load("seqeval")

api = HfApi()

ld = LanguageData.from_db()


u = urielplus.URIELPlus()
u.set_cache(True)
try:
    u.integrate_grambank()
except SystemExit:
    print("already using GramBank")
try:
    u.set_glottocodes()
except SystemExit:
    print("already using Glottocodes")


def get_available_adapters():
    # get all adapters from huggingface
    all_models = api.list_models(author="AdapterHub", library="adapter-transformers", search="xlm-roberta-base-")

    to_load = {
        m.modelId: m.modelId.split("xlm-roberta-base-")[-1].rsplit("-wiki_pfeiffer", 1)[0]
        for m in all_models
        if m.modelId.startswith("AdapterHub/xlm-roberta-base-") and m.modelId.endswith("-wiki_pfeiffer")
    }
    return to_load


def get_glots(to_load):
    manuals = {
        "Arabic": "arab1267",
        "Swahili": "swah1253",
        "Bengali": "beng1282",
        "Chinese": "mand1415",
        "Persian": "west2369",
        "Yoruba": "ilaa1246",
        "Nepali": "nepa1254",
        "Quechua": "cusc1236",
        "Estonian": "esto1258",
        "Guarani": "east2555",
    }

    glots = {}
    probs = []

    for lang in to_load.values():
        eng = ld.get(lang, tag_type=TagType.BCP_47_CODE).english_name
        glot = ld.get(lang, tag_type=TagType.BCP_47_CODE).glottocode
        # we need to find if glot is in distances
        if not glot:
            if eng in manuals.keys():
                glot = manuals[eng]
        if eng and glot:
            glots[lang] = glot
        else:
            probs.append(lang)
    if probs:
        print("no glottocodes found for these languages: ", probs)
        print("removing them from further consideration")
    for prob in probs:
        del to_load[prob]
        # happens in-place
    return glots


def merge_loaded_adapters(
    model, merge_adapter_name="joined_adapter", weights=None, delete_other=False, patterns=False, model_type="roberta"
):
    # to ensure we don't get problems, we check the config of all adapters
    all_adapters = list(model.adapters_config.adapters.keys())
    config_id = model.adapters_config.adapters[all_adapters[0]]
    config = model.adapters_config.config_map[config_id]

    for i in range(1, len(all_adapters)):
        config_id = model.adapters_config.adapters[all_adapters[i]]
        config_i = model.adapters_config.config_map[config_id]
        assert config == config_i, (
            f"Config mismatch: {config} vs {config_i}\nCurrent methodology only works for same config"
        )

    if not weights:
        weights = [1 / len(all_adapters)] * len(all_adapters)
    if len(weights) != len(all_adapters):
        raise ValueError(f"Weights length {len(weights)} does not match number of adapters {len(all_adapters)}")

    if not patterns:
        patterns = [
            f"{model_type}\.encoder\.layer\.([\d\w]+)\.output\.adapters\.(?:\w+)\.(\w+)(?:\.0)?\.(\w+)",
            f"{model_type}\.invertible_adapters\.(?:\w+)\.(\w+)\.(\d)\.(\w+)",
        ]
    comp_patterns = [re.compile(pattern) for pattern in patterns]
    organized_layers = {}
    for i, pattern in enumerate(patterns):
        # we make a dictionary for each pattern
        organized_layers[i] = {}

    for key in model.state_dict().keys():
        for i, pattern in enumerate(comp_patterns):
            match = re.search(pattern, key)
            if match:
                one = match.group(1)
                two = match.group(2)
                three = match.group(3)
                if one not in organized_layers[i]:
                    organized_layers[i][one] = {}
                if two not in organized_layers[i][one]:
                    organized_layers[i][one][two] = {}
                if three not in organized_layers[i][one][two]:
                    organized_layers[i][one][two][three] = []
                organized_layers[i][one][two][three].append(key)

    new_state_dict = OrderedDict()
    sd = model.state_dict()

    for i, one in organized_layers.items():
        for one, two in one.items():
            for two, three in two.items():
                for three, keys in three.items():
                    result = sum([sd[key] * weights[j] for j, key in enumerate(keys)])
                    if two == "adapter_down":
                        new_state_dict[
                            f"{model_type}.encoder.layer.{one}.output.adapters.{merge_adapter_name}.{two}.0.{three}"
                        ] = result
                    elif two == "adapter_up":
                        new_state_dict[
                            f"{model_type}.encoder.layer.{one}.output.adapters.{merge_adapter_name}.{two}.{three}"
                        ] = result
                    else:
                        # we are in the second pattern
                        new_state_dict[f"{model_type}.invertible_adapters.{merge_adapter_name}.{one}.{two}.{three}"] = (
                            result
                        )

    # we now load in the new model
    if merge_adapter_name in model.adapters_config.adapters.keys():
        # remove the old one
        model.delete_adapter(merge_adapter_name)
    model.add_adapter(merge_adapter_name, config=config)
    for name, param in model.named_parameters():
        # e.g. "roberta.encoder.layer.0.output.adapters.joined_adapter.adapter_down.0.weight"
        if merge_adapter_name in name and name in new_state_dict:
            param.data.copy_(new_state_dict[name])
    if delete_other:
        for key in list(model.adapters_config.adapters.keys()):
            if key != merge_adapter_name:
                model.delete_adapter(key)

    # no need to return anything as the model is changed in place


def typological_approximation(target, glots, distance_type):
    """
    This function takes a target language and a list of languages.
    It weights the other languages depending on their closeness to the target language.
    "distance_type" can be of following:
    "featural", "syntactic", "phonological", "geographic", "genetic", "morphological", "inventory"
    """

    # 1. retrieve closeness score of all languages to target language
    weights = []
    for lang, glot in glots.items():
        # get the distance
        try:
            dist = u.new_distance(distance_type, [glot, target])
            # print(f"Distance {lang} to {target}: {dist}")
        except Exception as e:
            # print(f"Error: {lang} - {glot} - {target}")
            print(e)
            dist = 0
        weights.append(dist)

    # 1. softmax over weights
    # print(f"Weights before softmax: {weights}")
    weights = torch.softmax(torch.tensor(weights), dim=0)
    # we need to convert to list
    weights = weights.tolist()
    # print(f"Weights after softmax: {weights}")
    return weights
