import gvar as gv
import numpy as np
import pytest

from latcorr.ground_state import (
    pt3_ratio_fit,
    pt3_ratio_im_fcn,
    pt3_ratio_re_fcn,
    pt3_ratio_two_state_fit,
)


def _two_state_params():
    return {
        "E0": 0.45,
        "dE1": 0.35,
        "z0": 1.1,
        "z1": 0.45,
        "O00_re": 0.8,
        "O01_re": 0.25,
        "O11_re": 0.12,
        "O00_im": 0.2,
        "O01_im": 0.08,
        "O11_im": 0.04,
    }


def _two_state_prior(params: dict) -> gv.BufferDict:
    prior = gv.BufferDict()
    prior["E0"] = gv.gvar(params["E0"], 0.15)
    prior["log(dE1)"] = gv.gvar(np.log(params["dE1"]), 0.25)
    for key in [
        "z0",
        "z1",
        "O00_re",
        "O01_re",
        "O11_re",
        "O00_im",
        "O01_im",
        "O11_im",
    ]:
        prior[key] = gv.gvar(params[key], abs(params[key]) * 0.25 + 0.05)
    return prior


def _mock_ratio_data(params: dict, tsep_ls: list[int], Lt: int):
    ratio_real: dict[int, np.ndarray] = {}
    ratio_imag: dict[int, np.ndarray] = {}
    for tsep in tsep_ls:
        real_row = np.empty(tsep + 1, dtype=object)
        imag_row = np.empty(tsep + 1, dtype=object)
        for tau in range(tsep + 1):
            real_mean = pt3_ratio_re_fcn(
                float(tsep), float(tau), params, Lt, nstate=2
            )
            imag_mean = pt3_ratio_im_fcn(
                float(tsep), float(tau), params, Lt, nstate=2
            )
            real_row[tau] = gv.gvar(float(np.asarray(real_mean).reshape(())), 5e-4)
            imag_row[tau] = gv.gvar(float(np.asarray(imag_mean).reshape(())), 5e-4)
        ratio_real[tsep] = real_row
        ratio_imag[tsep] = imag_row
    return ratio_real, ratio_imag


def test_pt3_ratio_fit_returns_lsqfit_result():
    Lt = 32
    tsep_ls = [8, 10]
    params = _two_state_params()
    ratio_real, ratio_imag = _mock_ratio_data(params, tsep_ls, Lt)

    fit = pt3_ratio_fit(
        tsep_ls,
        2,
        ratio_real,
        ratio_imag,
        Lt,
        prior=_two_state_prior(params),
        label="mock",
    )

    assert fit.dof > 0
    assert 0 <= fit.Q <= 1
    assert abs(fit.p["O00_re"].mean - params["O00_re"]) < 0.1
    assert abs(fit.p["O00_im"].mean - params["O00_im"]) < 0.1


def test_pt3_ratio_two_state_fit_accepts_pt2_fit_result_priors():
    Lt = 32
    tsep_ls = [8, 10]
    params = _two_state_params()
    ratio_real, ratio_imag = _mock_ratio_data(params, tsep_ls, Lt)
    pt2_prior = gv.BufferDict()
    pt2_prior["E0"] = gv.gvar(params["E0"], 0.01)
    pt2_prior["log(dE1)"] = gv.gvar(np.log(params["dE1"]), 0.01)
    pt2_prior["z0"] = gv.gvar(params["z0"], 0.01)
    pt2_prior["z1"] = gv.gvar(params["z1"], 0.01)

    class FitResult:
        p = pt2_prior

    fit = pt3_ratio_two_state_fit(
        tsep_ls,
        2,
        ratio_real,
        ratio_imag,
        Lt,
        prior=_two_state_prior(params),
        pt2_fit_res=FitResult(),
    )

    assert fit.dof > 0
    assert abs(fit.prior["E0"].mean - params["E0"]) < 1e-12
    assert abs(fit.prior["log(dE1)"].mean - np.log(params["dE1"])) < 1e-12
    assert abs(fit.prior["z0"].mean - params["z0"]) < 1e-12
    assert abs(fit.prior["z1"].mean - params["z1"]) < 1e-12


def test_pt3_ratio_fit_rejects_empty_tau_window():
    params = _two_state_params()
    ratio_real, ratio_imag = _mock_ratio_data(params, [4], 32)

    with pytest.raises(ValueError, match="empty tau fit window"):
        pt3_ratio_fit(
            [4],
            3,
            ratio_real,
            ratio_imag,
            32,
            prior=_two_state_prior(params),
        )


def test_pt3_ratio_fit_rejects_missing_tsep_key():
    params = _two_state_params()
    ratio_real, ratio_imag = _mock_ratio_data(params, [8], 32)

    with pytest.raises(KeyError, match="ratio_real is missing tsep 10"):
        pt3_ratio_fit(
            [8, 10],
            2,
            ratio_real,
            ratio_imag,
            32,
            prior=_two_state_prior(params),
        )


def test_pt3_ratio_fit_can_select_real_or_imag_part():
    Lt = 32
    tsep_ls = [8, 10]
    params = _two_state_params()
    ratio_real, ratio_imag = _mock_ratio_data(params, tsep_ls, Lt)
    prior = _two_state_prior(params)

    fit_re = pt3_ratio_fit(
        tsep_ls,
        2,
        ratio_real,
        ratio_imag,
        Lt,
        prior=prior,
        part="re",
    )
    fit_im = pt3_ratio_fit(
        tsep_ls,
        2,
        ratio_real,
        ratio_imag,
        Lt,
        prior=prior,
        part="im",
    )

    assert fit_re.dof > 0
    assert fit_im.dof > 0
