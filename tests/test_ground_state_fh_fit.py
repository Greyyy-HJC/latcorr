import gvar as gv
import numpy as np
import pytest

from latcorr.ground_state import (
    fh_fit,
    fh_im_fcn,
    fh_one_state_fit,
    fh_re_fcn,
    fh_two_state_fit,
)


def _two_state_params():
    return {
        "E0": 0.45,
        "dE1": 0.35,
        "O00_re": 0.8,
        "O00_im": 0.2,
        "sum_re_excited_coeff": 0.12,
        "sum_re_offset": 0.05,
        "sum_re_exp_offset": 0.04,
        "sum_im_excited_coeff": 0.10,
        "sum_im_offset": 0.03,
        "sum_im_exp_offset": 0.02,
        "sum_den_exp_coeff": 0.08,
    }


def _two_state_prior(params: dict) -> gv.BufferDict:
    prior = gv.BufferDict()
    prior["E0"] = gv.gvar(params["E0"], 0.15)
    prior["log(dE1)"] = gv.gvar(np.log(params["dE1"]), 0.25)
    for key in [
        "O00_re",
        "O00_im",
        "sum_re_excited_coeff",
        "sum_re_offset",
        "sum_re_exp_offset",
        "sum_im_excited_coeff",
        "sum_im_offset",
        "sum_im_exp_offset",
        "sum_den_exp_coeff",
    ]:
        prior[key] = gv.gvar(params[key], abs(params[key]) * 0.25 + 0.05)
    return prior


def _mock_fh_data(params: dict, tsep_ls: list[int], tau_cut: int):
    dt = tsep_ls[1] - tsep_ls[0]
    t = np.asarray(tsep_ls[:-1], dtype=float)
    real_mean = fh_re_fcn(t, tau_cut, params, nstate=2, dt=dt)
    imag_mean = fh_im_fcn(t, tau_cut, params, nstate=2, dt=dt)
    return gv.gvar(real_mean, np.full_like(real_mean, 5e-4)), gv.gvar(
        imag_mean, np.full_like(imag_mean, 5e-4)
    )


def test_fh_fit_returns_lsqfit_result():
    params = _two_state_params()
    tsep_ls = [6, 8, 10, 12]
    tau_cut = 2
    fh_real, fh_imag = _mock_fh_data(params, tsep_ls, tau_cut)

    fit = fh_fit(
        fh_real,
        fh_imag,
        tsep_ls,
        tau_cut,
        prior=_two_state_prior(params),
        label="mock",
    )

    assert fit.dof > 0
    assert 0 <= fit.Q <= 1
    assert abs(fit.p["O00_re"].mean - params["O00_re"]) < 0.1
    assert abs(fit.p["O00_im"].mean - params["O00_im"]) < 0.1


def test_fh_two_state_fit_accepts_pt2_fit_result_priors():
    params = _two_state_params()
    tsep_ls = [6, 8, 10, 12]
    tau_cut = 2
    fh_real, fh_imag = _mock_fh_data(params, tsep_ls, tau_cut)
    pt2_prior = gv.BufferDict()
    pt2_prior["E0"] = gv.gvar(params["E0"], 0.01)
    pt2_prior["log(dE1)"] = gv.gvar(np.log(params["dE1"]), 0.01)

    class FitResult:
        p = pt2_prior

    fit = fh_two_state_fit(
        fh_real,
        fh_imag,
        tsep_ls,
        tau_cut,
        prior=_two_state_prior(params),
        pt2_fit_res=FitResult(),
    )

    assert fit.dof > 0
    assert abs(fit.prior["E0"].mean - params["E0"]) < 1e-12
    assert abs(fit.prior["log(dE1)"].mean - np.log(params["dE1"])) < 1e-12


def test_fh_one_state_fit_returns_lsqfit_result():
    tsep_ls = [4, 5, 6]
    params = {"E0": 0.5, "O00_re": 0.8, "O00_im": 0.2}
    t = np.asarray(tsep_ls[:-1], dtype=float)
    fh_real = gv.gvar(fh_re_fcn(t, 0, params, nstate=1), [5e-4, 5e-4])
    fh_imag = gv.gvar(fh_im_fcn(t, 0, params, nstate=1), [5e-4, 5e-4])
    prior = gv.BufferDict()
    prior["E0"] = gv.gvar(params["E0"], 0.1)
    prior["O00_re"] = gv.gvar(params["O00_re"], 0.2)
    prior["O00_im"] = gv.gvar(params["O00_im"], 0.2)

    fit = fh_one_state_fit(fh_real, fh_imag, tsep_ls, prior=prior)

    assert fit.dof > 0
    assert 0 <= fit.Q <= 1


def test_fh_fit_rejects_bad_data_length():
    params = _two_state_params()
    fh_real, fh_imag = _mock_fh_data(params, [6, 8, 10], tau_cut=2)

    with pytest.raises(ValueError, match="fh_real length"):
        fh_fit(
            fh_real[:-1],
            fh_imag,
            [6, 8, 10],
            2,
            prior=_two_state_prior(params),
        )


def test_fh_fit_accepts_explicit_dt():
    params = _two_state_params()
    fh_real, fh_imag = _mock_fh_data(params, [6, 8, 10], tau_cut=2)

    fit = fh_fit(
        fh_real,
        fh_imag,
        [6, 8, 11],
        2,
        prior=_two_state_prior(params),
        dt=2,
    )

    assert fit.dof > 0


def test_fh_fit_can_select_real_or_imag_part():
    params = _two_state_params()
    tsep_ls = [6, 8, 10, 12]
    tau_cut = 2
    fh_real, fh_imag = _mock_fh_data(params, tsep_ls, tau_cut)
    prior = _two_state_prior(params)

    fit_re = fh_fit(
        fh_real,
        fh_imag,
        tsep_ls,
        tau_cut,
        prior=prior,
        part="re",
    )
    fit_im = fh_fit(
        fh_real,
        fh_imag,
        tsep_ls,
        tau_cut,
        prior=prior,
        part="im",
    )

    assert fit_re.dof > 0
    assert fit_im.dof > 0
