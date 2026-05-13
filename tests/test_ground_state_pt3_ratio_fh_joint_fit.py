import gvar as gv
import numpy as np

from latcorr.ground_state import (
    fh_im_fcn,
    fh_re_fcn,
    pt3_ratio_fh_joint_fit,
    pt3_ratio_im_fcn,
    pt3_ratio_re_fcn,
    pt3_ratio_two_state_fh_one_state_fit,
)


def _params():
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


def _prior(params: dict) -> gv.BufferDict:
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


def _mock_ratio(params: dict, tsep_ls: list[int], Lt: int):
    ratio_real: dict[int, np.ndarray] = {}
    ratio_imag: dict[int, np.ndarray] = {}
    for tsep in tsep_ls:
        real_row = np.empty(tsep + 1, dtype=object)
        imag_row = np.empty(tsep + 1, dtype=object)
        for tau in range(tsep + 1):
            real = pt3_ratio_re_fcn(float(tsep), float(tau), params, Lt, nstate=2)
            imag = pt3_ratio_im_fcn(float(tsep), float(tau), params, Lt, nstate=2)
            real_row[tau] = gv.gvar(float(np.asarray(real).reshape(())), 5e-4)
            imag_row[tau] = gv.gvar(float(np.asarray(imag).reshape(())), 5e-4)
        ratio_real[tsep] = real_row
        ratio_imag[tsep] = imag_row
    return ratio_real, ratio_imag


def _mock_fh(params: dict, tsep_ls: list[int]):
    t = np.asarray(tsep_ls[:-1], dtype=float)
    real = fh_re_fcn(t, tau_cut=0, p=params, nstate=1)
    imag = fh_im_fcn(t, tau_cut=0, p=params, nstate=1)
    return gv.gvar(real, np.full_like(real, 5e-4)), gv.gvar(
        imag, np.full_like(imag, 5e-4)
    )


def test_pt3_ratio_fh_joint_fit_returns_lsqfit_result():
    Lt = 32
    tsep_ls = [8, 10, 12]
    tau_cut = 2
    params = _params()
    ratio_real, ratio_imag = _mock_ratio(params, tsep_ls, Lt)
    fh_real, fh_imag = _mock_fh(params, tsep_ls)

    fit = pt3_ratio_fh_joint_fit(
        tsep_ls,
        tau_cut,
        ratio_real,
        ratio_imag,
        fh_real,
        fh_imag,
        Lt,
        prior=_prior(params),
        label="mock",
    )

    assert fit.dof > 0
    assert 0 <= fit.Q <= 1
    assert abs(fit.p["O00_re"].mean - params["O00_re"]) < 0.1
    assert abs(fit.p["O00_im"].mean - params["O00_im"]) < 0.1


def test_pt3_ratio_two_state_fh_one_state_fit_accepts_pt2_fit_result():
    Lt = 32
    tsep_ls = [8, 10, 12]
    tau_cut = 2
    params = _params()
    ratio_real, ratio_imag = _mock_ratio(params, tsep_ls, Lt)
    fh_real, fh_imag = _mock_fh(params, tsep_ls)
    pt2_prior = gv.BufferDict()
    pt2_prior["E0"] = gv.gvar(params["E0"], 0.01)
    pt2_prior["log(dE1)"] = gv.gvar(np.log(params["dE1"]), 0.01)
    pt2_prior["z0"] = gv.gvar(params["z0"], 0.01)
    pt2_prior["z1"] = gv.gvar(params["z1"], 0.01)

    class FitResult:
        p = pt2_prior

    fit = pt3_ratio_two_state_fh_one_state_fit(
        tsep_ls,
        tau_cut,
        ratio_real,
        ratio_imag,
        fh_real,
        fh_imag,
        Lt,
        prior=_prior(params),
        pt2_fit_res=FitResult(),
    )

    assert fit.dof > 0
    assert abs(fit.prior["E0"].mean - params["E0"]) < 1e-12
    assert abs(fit.prior["log(dE1)"].mean - np.log(params["dE1"])) < 1e-12
    assert abs(fit.prior["z0"].mean - params["z0"]) < 1e-12
    assert abs(fit.prior["z1"].mean - params["z1"]) < 1e-12


def test_pt3_ratio_fh_joint_fit_can_select_real_or_imag_part():
    Lt = 32
    tsep_ls = [8, 10, 12]
    tau_cut = 2
    params = _params()
    ratio_real, ratio_imag = _mock_ratio(params, tsep_ls, Lt)
    fh_real, fh_imag = _mock_fh(params, tsep_ls)
    prior = _prior(params)

    fit_re = pt3_ratio_fh_joint_fit(
        tsep_ls,
        tau_cut,
        ratio_real,
        ratio_imag,
        fh_real,
        fh_imag,
        Lt,
        prior=prior,
        part="re",
    )
    fit_im = pt3_ratio_fh_joint_fit(
        tsep_ls,
        tau_cut,
        ratio_real,
        ratio_imag,
        fh_real,
        fh_imag,
        Lt,
        prior=prior,
        part="im",
    )

    assert fit_re.dof > 0
    assert fit_im.dof > 0
