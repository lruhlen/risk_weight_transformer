import os

import pandas as pd
import pytest
import numpy as np

from risk_weight_transformer import RiskWeightTransformer


@pytest.fixture
def resources_directory():
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "resources"))


@pytest.fixture
def risk_weight_data(resources_directory):
    fname = os.path.join(resources_directory, "data_test_RiskWeightTransformer.csv")
    df = pd.read_csv(fname)
    return df


def test_RiskWeightTransformer(risk_weight_data):

    feature_names = ["animal", "planet"]
    target_name = "is_good"
    rwt = RiskWeightTransformer(feature_names, target_name=target_name)
    rwt.fit(risk_weight_data)
    # pylint: disable=protected-access
    assert rwt._risk_weight_lookup_dicts, "Risk weight lookup dictionary was not populated"
    assert rwt._risk_weight_lookup_dicts["animal"]["moose"] == 1.0
    assert rwt._risk_weight_lookup_dicts["animal"]["squirrel"] == 1.0 / 3.0
    assert rwt._risk_weight_lookup_dicts["animal"]["unknown"] == 2.0 / 3.0
    assert rwt._risk_weight_lookup_dicts["animal"][None] == 2.0 / 3.0
    assert rwt._risk_weight_lookup_dicts["planet"]["mars"] == 0.5
    assert rwt._risk_weight_lookup_dicts["planet"]["jupiter"] == 0.5
    assert rwt._risk_weight_lookup_dicts["planet"]["venus"] == 1.0
    assert rwt._risk_weight_lookup_dicts["planet"]["unknown"] == 2.0 / 3.0
    assert rwt._risk_weight_lookup_dicts["planet"][None] == 2.0 / 3.0

    tmp_df = risk_weight_data[feature_names]
    tmp = rwt.transform(tmp_df)
    assert (tmp[:, 0] == np.array([1.0, 1.0, 1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0])).all()
    assert (tmp[:, 1] == np.array([0.5, 0.5, 0.5, 0.5, 1.0])).all()

    tmp_df = pd.DataFrame(data=[["gryphon", "mercury"]], columns=["animal", "planet"])
    tmp = rwt.transform(tmp_df)
    assert (tmp == np.array([2.0 / 3.0, 2.0 / 3.0])).all()


#TODO: test transformer works within an sklearn Pipeline
def test_pipeline_integration():
    assert False
