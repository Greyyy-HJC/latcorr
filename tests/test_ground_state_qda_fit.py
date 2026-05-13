import gvar as gv
import numpy as np

from latcorr.ground_state import (
    pt2_re_fcn,
    qda_fit,
    qda_im_fcn,
    qda_joint_fit,
    qda_re_fcn,
    qda_two_state_fit,
    qda_two_state_joint_fit,
)


def _params():
    return {
        "E0": 0.45,
        "dE1": 0.35,
        "z0": 1.1,
        "z1": 0.45,
        "O00_re": 0.8,
        "O01_re": 0.25,
        "O00_im": 0.2,
        "O01_im": 0.08,
    }


def _prior(params: dict) -> gv.BufferDict:
    prior = gv.BufferDict()
    prior["E0"] = gv.gvar(params["E0"], 0.15)
    prior["log(dE1)"] = gv.gvar(np.log(params["dE1"]), 0.25)
    for key in ["z0", "z1", "O00_re", "O01_re", "O00_im", "O01_im"]:
        prior[key] = gv.gvar(params[key], abs(params[key]) * 0.25 + 0.05)
    return prior


def _mock_data(params: dict, Lt: int):
    t = np.arange(Lt)
    pt2 = pt2_re_fcn(t, params, Lt, nstate=2)
    qda_real = qda_re_fcn(t, params, Lt, nstate=2)
    qda_imag = qda_im_fcn(t, params, Lt, nstate=2)
    return (
        gv.gvar(pt2, np.maximum(0.02 * pt2, 1e-6)),
        gv.gvar(qda_real, np.maximum(0.02 * np.abs(qda_real), 1e-6)),
        gv.gvar(qda_imag, np.maximum(0.02 * np.abs(qda_imag), 1e-6)),
    )


def test_qda_fit_returns_lsqfit_result():
    Lt = 32
    params = _params()
    _, qda_real, qda_imag = _mock_data(params, Lt)

    fit = qda_fit(
        qda_real,
        qda_imag,
        3,
        12,
        Lt,
        prior=_prior(params),
        label="mock",
    )

    assert fit.dof > 0
    assert 0 <= fit.Q <= 1
    assert abs(fit.p["O00_re"].mean - params["O00_re"]) < 0.1
    assert abs(fit.p["O00_im"].mean - params["O00_im"]) < 0.1


def test_qda_two_state_fit_uses_pt2_fit_result_as_wider_priors():
    Lt = 32
    params = _params()
    _, qda_real, qda_imag = _mock_data(params, Lt)
    pt2_prior = gv.BufferDict()
    pt2_prior["E0"] = gv.gvar(params["E0"], 0.01)
    pt2_prior["log(dE1)"] = gv.gvar(np.log(params["dE1"]), 0.02)
    pt2_prior["z0"] = gv.gvar(params["z0"], 0.03)
    pt2_prior["z1"] = gv.gvar(params["z1"], 0.04)

    class FitResult:
        p = pt2_prior

    fit = qda_two_state_fit(
        qda_real,
        qda_imag,
        3,
        12,
        Lt,
        prior=_prior(params),
        pt2_fit_res=FitResult(),
    )

    assert abs(fit.prior["E0"].mean - params["E0"]) < 1e-12
    assert abs(fit.prior["E0"].sdev - 0.05) < 1e-12
    assert abs(fit.prior["log(dE1)"].sdev - 0.10) < 1e-12
    assert abs(fit.prior["z0"].sdev - 0.15) < 1e-12
    assert abs(fit.prior["z1"].sdev - 0.20) < 1e-12


def test_qda_joint_fit_returns_lsqfit_result():
    Lt = 32
    params = _params()
    pt2_avg, qda_real, qda_imag = _mock_data(params, Lt)

    fit = qda_joint_fit(
        pt2_avg,
        qda_real,
        qda_imag,
        np.arange(3, 12),
        np.arange(3, 12),
        Lt,
        prior=_prior(params),
    )

    assert fit.dof > 0
    assert 0 <= fit.Q <= 1


def test_qda_two_state_joint_fit_wrapper():
    Lt = 32
    params = _params()
    pt2_avg, qda_real, qda_imag = _mock_data(params, Lt)

    fit = qda_two_state_joint_fit(
        pt2_avg,
        qda_real,
        qda_imag,
        np.arange(3, 12),
        np.arange(3, 12),
        Lt,
        prior=_prior(params),
    )

    assert fit.dof > 0


def test_qda_fit_and_joint_fit_can_select_real_or_imag_part():
    Lt = 32
    params = _params()
    pt2_avg, qda_real, qda_imag = _mock_data(params, Lt)
    prior = _prior(params)

    fit_re = qda_fit(
        qda_real,
        qda_imag,
        3,
        12,
        Lt,
        prior=prior,
        part="re",
    )
    fit_im = qda_joint_fit(
        pt2_avg,
        qda_real,
        qda_imag,
        np.arange(3, 12),
        np.arange(3, 12),
        Lt,
        prior=prior,
        part="im",
    )

    assert fit_re.dof > 0
    assert fit_im.dof > 0
