"""Tests for FairPredictor"""

import pandas as pd
import sklearn.ensemble
import sklearn.tree
import oxonfair as fair
import numpy as np
from oxonfair import FairPredictor
from oxonfair.utils import group_metrics as gm

PLT_EXISTS = True
try:
    import matplotlib.pyplot as plt
    plt.title
except ModuleNotFoundError:
    PLT_EXISTS = False

classifier_type = sklearn.ensemble.RandomForestClassifier

train_data = pd.read_csv("https://autogluon.s3.amazonaws.com/datasets/Inc/train.csv")
test_data = pd.read_csv("https://autogluon.s3.amazonaws.com/datasets/Inc/test.csv")

# Merge and shuffle the data
total_data = pd.concat([train_data, test_data])
y = total_data["class"] == " <=50K"
total_data = total_data.drop(columns="class")
total_data = pd.get_dummies(total_data)


def resample():
    global train, train_y, val, val_y, test, test_y, predictor
    global train_dict, val_dict, test_dict
    global train_dict_g, val_dict_g, test_dict_g
    train = total_data.sample(frac=0.5)
    val_test = total_data.drop(train.index)
    train_y = y.iloc[train.index]
    val_test_y = y.drop(train_y.index)
    val = val_test.sample(frac=0.4)
    test = val_test.drop(val.index)
    val_y = y.iloc[val.index]
    test_y = val_test_y.drop(val.index)

    predictor = classifier_type()
    predictor.fit(train, train_y)

    train_dict = {"data": train, "target": train_y}
    val_dict = {"data": val, "target": val_y}
    test_dict = {"data": test, "target": test_y}

    val_dict_g = fair.DataDict(val_y, val, val['sex_ Female'])
    test_dict_g = fair.DataDict(test_y, test, test['sex_ Female'])


resample()


def test_base_functionality(val_dict=val_dict, test_dict=test_dict):
    "not calling fit should not alter predict or predict_proba"
    if 'groups' in val_dict:
        fpredictor = FairPredictor(predictor, val_dict)
    else:
        fpredictor = FairPredictor(predictor, val_dict, "sex_ Female")
    e1 = fpredictor.evaluate(val_dict)
    e2 = fpredictor.evaluate()
    assert (e1 == e2).all().all()
    f1 = fpredictor.evaluate_fairness()
    f2 = fpredictor.evaluate_fairness(val_dict)
    assert (f1 == f2).all().all()
    g1 = fpredictor.evaluate_groups(val_dict)
    g2 = fpredictor.evaluate_groups()
    assert (g1 == g2).all().all()
    proba = fpredictor.predict_proba(val_dict)
    pred = fpredictor.predict(val_dict)
    if 'groups' not in val_dict:
        proba2 = predictor.predict_proba(val_dict["data"])
        assert (proba == proba2).all().all()
        assert (pred == predictor.predict(val_dict["data"])).all().all()

    fpredictor.evaluate(verbose=False)
    fpredictor.evaluate_fairness(verbose=False)
    fpredictor.evaluate_groups(verbose=False)
    fpredictor.evaluate_groups(verbose=False, return_original=False)
    fpredictor.evaluate_groups(return_original=False)

    fpredictor.evaluate_groups(test_dict)


def test_conflict_groups():
    """When we overwrite existing groups calling fair predictor this should preserve metrics when they match,
    and otherwise alter them"""
    fpred = FairPredictor(predictor, val_dict_g)
    fpred2 = FairPredictor(predictor, val_dict_g, 'sex_ Female')
    fpred3 = FairPredictor(predictor, val_dict_g, 'race_ White')

    assert (fpred.evaluate_fairness() == fpred2.evaluate_fairness()).all().all()
    assert (fpred2.evaluate_fairness() == fpred2.evaluate_fairness(val_dict_g)).all().all()
    assert (fpred3.evaluate_fairness() == fpred3.evaluate_fairness(val_dict_g)).all().all()

    assert (fpred2.evaluate_fairness() != fpred3.evaluate_fairness(val_dict_g)).any().any()


def test_fit_creates_updated(use_fast=True):
    """eval should return 'updated' iff fit has been called"""
    fpredictor = FairPredictor(predictor, val_dict, use_fast=use_fast)
    assert isinstance(fpredictor.evaluate(), pd.Series)
    fpredictor.fit(gm.accuracy, gm.recall, 0)  # constraint is intentionally slack
    assert not isinstance(fpredictor.evaluate(), pd.Series)
    assert 'updated' in fpredictor.evaluate().columns


def test_fit_creates_updated_slow():
    test_fit_creates_updated(False)


def test_fit_creates_updated_hybrid():
    test_fit_creates_updated('hybrid')


def test_base_with_groups():
    'Test base functionality holds when groups are provided'
    test_base_functionality(val_dict_g, test_dict_g)


def test_implicit_groups():
    "try without using any value for groups"
    val2 = val_dict.copy()
    val2["groups"] = val_dict["data"]["sex_ Female"]
    test2 = test_dict.copy()
    test2["groups"] = test_dict["data"]["sex_ Female"]
    fpredictor = FairPredictor(predictor, val2)
    fpredictor.evaluate_groups(verbose=True)
    fpredictor.evaluate_groups(test2, verbose=True)
    fpredictor.fit(gm.accuracy, gm.accuracy_parity, 0.01)
    if PLT_EXISTS:
        fpredictor.plot_frontier()
    fpredictor.evaluate_groups(verbose=True)
    fpredictor.evaluate_groups(test2, verbose=True)


def test_no_groups(use_fast=True):
    "check pathway works with no groups"
    fairp = fair.FairPredictor(predictor, val_dict, use_fast=use_fast)
    fairp.evaluate()
    fairp.evaluate_groups()
    fairp.evaluate_fairness()
    assert (fairp.predict_proba(val_dict) == predictor.predict_proba(val)).all().all()
    assert (fairp.predict(val_dict) == predictor.predict(val)).all().all()
    fairp.fit(gm.accuracy, gm.f1, 0)
    if PLT_EXISTS:
        fairp.plot_frontier()
    fairp.evaluate(test_dict)
    fairp.evaluate_fairness(test_dict)
    fairp.evaluate_groups(test_dict)
    if PLT_EXISTS:
        fairp.plot_frontier(test_dict)


def test_groups(use_fast=True):
    "Check that evaluate correctly passes groups to fairness metrics"
    fairp = fair.FairPredictor(predictor, val_dict, groups="sex_ Female", use_fast=use_fast)
    assert fairp.evaluate(metrics={'EO': gm.equal_opportunity}).to_numpy().any()


def test_predict(use_fast=True):
    "check that fairpredictor returns the same as a standard predictor before fit is called"
    fpredictor = fair.FairPredictor(
        predictor, test_dict, groups="sex_ Female", use_fast=use_fast
    )
    assert (predictor.predict(test) == fpredictor.predict(test_dict)).all()
    assert (predictor.predict_proba(test) == fpredictor.predict_proba(test_dict)).all()


def test_pathologoical(use_fast=True):
    "Returns a single constant classifier"
    fpredictor = fair.FairPredictor(predictor, val_dict, groups="sex_ Female", use_fast=False)
    fpredictor.fit(gm.roc_auc, gm.equalized_odds, 0.75)
    fpredictor.plot_frontier()
    fpredictor.evaluate_fairness()


def test_pathologoical2(use_fast=True):
    "pass it the same objective twice"
    fpredictor = fair.FairPredictor(
        predictor, val_dict, groups="sex_ Female", use_fast=use_fast
    )
    fpredictor.fit(gm.balanced_accuracy, gm.balanced_accuracy, 0)
    if PLT_EXISTS:
        fpredictor.plot_frontier()
    fpredictor.evaluate_fairness()
    if PLT_EXISTS:
        fpredictor.plot_frontier(test_dict)
    fpredictor.evaluate_fairness(test_dict)


def test_recall_diff(use_fast=True):
    """Maximize accuracy while enforcing weak equal opportunity,
    such that the difference in recall between groups is less than 2.5%
    This also tests the sign functionality on constraints and the objective"""

    fpredictor = fair.FairPredictor(
        predictor, test_dict, "sex_ Female", use_fast=use_fast
    )

    fpredictor.fit(gm.accuracy, gm.recall.diff, 0.025)

    # Evaluate the change in fairness (recall difference corresponds to EO)
    measures = fpredictor.evaluate_fairness(verbose=False)

    assert measures["original"]["recall.diff"] > 0.025

    assert measures["updated"]["recall.diff"] <= 0.025
    measures = fpredictor.evaluate(verbose=False)
    acc = measures["updated"]["accuracy"]
    fpredictor.fit(gm.accuracy, gm.recall.diff, 0.025, greater_is_better_const=True)
    measures = fpredictor.evaluate_fairness(verbose=False)
    assert measures["original"]["recall.diff"] > 0.025

    fpredictor.fit(gm.accuracy, gm.recall.diff, 0.025, greater_is_better_obj=False)
    assert acc > fpredictor.evaluate(verbose=False)["updated"]["accuracy"]


def test_disp_impact(use_fast=True):
    """Enforce the 4/5 rule that the worst ratio between the proportion
      of positive decisions is greater than 0.9"""
    fpredictor = fair.FairPredictor(predictor, test_dict, "sex_ Female", use_fast=use_fast)
    fpredictor.fit(gm.accuracy, gm.disparate_impact, 0.9)

    measures = fpredictor.evaluate_fairness(metrics=gm.clarify_metrics, verbose=False)

    assert measures["original"]["disparate_impact"] < 0.9

    assert measures["updated"]["disparate_impact"] > 0.9


def test_min_recall(use_fast=True):
    "check that we can force recall >0.5 for all groups"
    fpredictor = fair.FairPredictor(
        predictor, test_dict, "sex_ Female", use_fast=use_fast
    )
    # Enforce that every group has a recall over 0.5
    fpredictor.fit(gm.accuracy, gm.recall.min, 0.5)
    scores = fpredictor.evaluate_groups(return_original=False, verbose=False)
    assert all(scores["recall"][:-1] > 0.5)


def test_no_groups_slow():
    "test slow pathway"
    test_no_groups(False)


def test_no_groups_hybrid():
    test_no_groups('hybrid')


def test_predict_slow():
    "test slow pathway"
    test_predict(False)


def test_predict_hybrid():
    test_predict('hybrid')


def test_pathologoical2_slow():
    "test slow pathway"
    test_pathologoical2(False)


def test_pathologoical2_hybrid():
    test_pathologoical2('hybrid')


def test_recall_diff_slow():
    "test slow pathway"
    test_recall_diff(False)


def test_recall_diff_hybrid():
    test_recall_diff('hybrid')


""" too slow -- keep disabled unless hunting non-deterministic bugs.
def test_many_recall_diff_hybrid(many=2000):
    count = 0
    for i in range(many):
        resample()
        try:
            test_recall_diff_hybrid()
        except:
            count += 1
    assert count == 0


def test_many_recall_diff_slow(many=200):
    count = 0
    for i in range(many):
        resample()
        try:
            test_recall_diff_slow()
        except:
            count += 1
    assert count == 0
# """


def test_min_recall_slow():
    "test slow pathway"
    test_min_recall(False)


def test_min_recall_hybrid():
    test_min_recall('hybrid')


def test_disp_impact_slow():
    "test slow pathway"
    test_disp_impact(False)


def test_disp_impact_hybrid():
    test_disp_impact('hybrid')


def test_recall_diff_inferred(use_fast=True):
    "use infered attributes instead of provided attributes"
    # train two new classifiers one to predict class without using sex and one to fpredict sex without using class
    def move_to_groups(my_dict, key, drop):
        new_dict = my_dict.copy()
        new_dict['groups'] = new_dict['data'][key]
        new_dict['data'] = new_dict['data'].drop(key, axis=1)
        new_dict['data'] = new_dict['data'].drop(drop, axis=1)
        return new_dict

    new_train = move_to_groups(train_dict, "sex_ Female", 'sex_ Male')
    new_val = move_to_groups(val_dict, "sex_ Female", 'sex_ Male')
    predictor = classifier_type()
    predictor.fit(new_train['data'], new_train['target'])
    protected = classifier_type()
    protected.fit(new_train['data'], new_train['groups'])
    # Build fair object using this and evaluate fairness n.b. classifier
    # accuracy decreases due to lack of access to the protected attribute, but
    # otherwise code is doing the same thing
    fpredictor = fair.FairPredictor(predictor, new_val, inferred_groups=protected, use_fast=use_fast)

    # Enforce that the new classifier will satisfy equalised odds (recall
    # difference between protected attributes of less than 2.5%) despite not
    # using sex at run-time

    fpredictor.fit(gm.accuracy, gm.recall.diff, 0.025)

    measures = fpredictor.evaluate_fairness(verbose=False)

    assert measures['original']['recall.diff'] > 0.025

    assert measures['updated']['recall.diff'] < 0.025

    # Prove that sex isn't being used by dropping it and reevaluating.

    fpredictor.evaluate_groups(new_train)
    # No test needed, code just has to run with sex dropped
    # predict should also work when run on raw data
    fpredictor.predict(new_train['data'])


def test_recall_diff_inferred_slow():
    test_recall_diff_inferred(False)


def test_recall_diff_inferred_hybrid():
    test_recall_diff_inferred('hybrid')


def test_normalized_classifier(fast=True):
    fpred = FairPredictor(predictor, val_dict, 'sex_ Female', use_fast=fast)
    fpred.fit(gm.accuracy, gm.demographic_parity, 0.02)
    response = fpred.predict_proba(test_dict['data'], force_normalization=True)
    assert (response >= 0).all().all()
    assert np.isclose(response.sum(1), 1).all()

    response2 = fpred.predict_proba(test_dict['data'])
    assert (np.argmax(response, 1) == np.argmax(response2, 1)).all()


def test_normalized_classifier_slow():
    test_normalized_classifier(False)


def test_normalized_classifier_hybrid():
    test_normalized_classifier('hybrid')


def test_total_metrics(fast=True):
    fpred = FairPredictor(predictor, val_dict, 'sex_ Female', use_fast=fast)
    fpred.evaluate(metrics={'a': gm.pos_data_proportion.max_diff})
    fpred.fit(gm.accuracy, gm.pos_pred_proportion.max, 0.02, greater_is_better_const=False)
    gm.pos_pred_proportion.max(val_dict_g['target'], val_dict_g['target'], val_dict_g['groups'])
    gm.directed_bias_amplification.per_group(val_dict_g['target'],
                                             np.random.standard_normal(val_dict_g['target'].shape), val_dict_g['groups'])
    fpred.fit(gm.accuracy, gm.directed_bias_amplification.max, 0.01, greater_is_better_const=False)
    fpred.fit(gm.accuracy, gm.bias_amplification.average, 0.01, greater_is_better_const=False)
    gm.neg_pred_proportion.per_group(val_dict_g['target'], val_dict_g['target'], val_dict_g['groups'])


def test_total_metrics_hybrid():
    test_total_metrics(fast='hybrid')


def test_total_metrics_slow():
    test_total_metrics(fast=False)


def test_frontier(use_fast=True):
    fpredictor = fair.FairPredictor(predictor, test_dict, "sex_ Female", use_fast=use_fast)
    fpredictor.fit(gm.accuracy, gm.recall.diff, 0.9)
    assert (fpredictor.frontier_scores() == fpredictor.frontier_scores(test_dict)).all()


def test_frontier_slow():
    test_frontier(False)


def test_frontier_hybrid():
    test_frontier('hybrid')


def test_selection_rate_diff_levelling_up(use_fast=True):
    """Maximize accuracy while enforcing demographic parity with levelling up."""

    fpredictor = fair.FairPredictor(
        predictor, test_dict, "sex_ Female", use_fast=use_fast)

    fpredictor.fit(gm.accuracy, gm.pos_pred_rate.diff, 0.025, force_levelling_up=True)

    rate = fpredictor.evaluate_groups(metrics={1: gm.pos_pred_rate}, verbose=False)[1]
    assert (rate['updated'].drop('Maximum difference') >= rate['original'].drop('Maximum difference')).all()
    assert (fpredictor.evaluate_groups() == fpredictor.evaluate_groups(test_dict)).all().all()
    # Evaluate the change in fairness (recall difference corresponds to EO)
    measures = fpredictor.evaluate_fairness(verbose=False)

    assert measures["original"]["statistical_parity"] > 0.025

    assert measures["updated"]["statistical_parity"] <= 0.025

    fpredictor.fit(gm.accuracy, gm.pos_pred_rate.diff, 0.025, force_levelling_up='-')
    rate = fpredictor.evaluate_groups(metrics={1: gm.pos_pred_rate}, verbose=False)[1]
    assert (rate['updated'].drop('Maximum difference') <= rate['original'].drop('Maximum difference')).all()
    # Evaluate the change in fairness (recall difference corresponds to EO)
    measures = fpredictor.evaluate_fairness(verbose=False)

    # Evaluate the change in fairness (recall difference corresponds to EO)
    measures = fpredictor.evaluate_fairness(verbose=False)

    assert measures["original"]["statistical_parity"] > 0.025

    assert measures["updated"]["statistical_parity"] <= 0.025


def test_selection_rate_diff_levelling_up_slow():
    test_selection_rate_diff_levelling_up(use_fast=False)


def test_selection_rate_diff_levelling_up_hybrid():
    test_selection_rate_diff_levelling_up(use_fast='hybrid')
