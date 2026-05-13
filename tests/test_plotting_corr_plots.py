import gvar as gv
import matplotlib.pyplot as plt
import numpy as np

from latcorr.ground_state import pt2_fit, pt2_re_fcn
from latcorr.plotting import (
    ff_joint_plot,
    ff_ratio_plot,
    ff_sum_plot,
    fh_plot,
    pt2_plot,
    pt3_ratio_plot,
    qda_ratio_plot,
)


class FitResult:
    def __init__(self, p):
        self.p = p


def test_qda_ratio_plot_draws_real_and_imag_data():
    trange = np.arange(3)
    qda_ratio_real = gv.gvar([1.0, 1.2, 1.5], [0.1, 0.1, 0.2])
    qda_ratio_imag = gv.gvar([0.2, 0.1, -0.1], [0.05, 0.05, 0.06])

    (fig_real, ax_real), (fig_imag, ax_imag) = qda_ratio_plot(
        trange,
        qda_ratio_real,
        qda_ratio_imag,
        id_label={"pz": 0, "z": 1},
    )

    assert ax_real.get_xlabel()
    assert ax_imag.get_xlabel()
    assert "qDA" in ax_real.get_ylabel()
    assert "qDA" in ax_imag.get_ylabel()

    plt.close(fig_real)
    plt.close(fig_imag)


def test_qda_ratio_plot_draws_fit_overlay():
    trange = np.arange(3)
    qda_ratio_real = gv.gvar([0.8, 0.82, 0.84], [0.05, 0.05, 0.05])
    qda_ratio_imag = gv.gvar([0.1, 0.11, 0.12], [0.02, 0.02, 0.02])
    qda_params = {
        "E0": gv.gvar(0.4, 0.02),
        "dE1": gv.gvar(0.3, 0.02),
        "z0": gv.gvar(1.1, 0.05),
        "z1": gv.gvar(0.4, 0.04),
        "O00_re": gv.gvar(0.8, 0.05),
        "O01_re": gv.gvar(0.2, 0.04),
        "O00_im": gv.gvar(0.1, 0.02),
        "O01_im": gv.gvar(0.04, 0.01),
    }
    pt2_params = {
        "E0": gv.gvar(0.4, 0.02),
        "dE1": gv.gvar(0.3, 0.02),
        "z0": gv.gvar(1.1, 0.05),
        "z1": gv.gvar(0.4, 0.04),
    }

    (fig_real, ax_real), (fig_imag, ax_imag) = qda_ratio_plot(
        trange,
        qda_ratio_real,
        qda_ratio_imag,
        fit_result=FitResult(qda_params),
        pt2_fit_result=FitResult(pt2_params),
        Lt=32,
    )

    assert ax_real.get_legend() is not None
    assert ax_imag.get_legend() is not None
    assert len(ax_real.collections) >= 2
    assert len(ax_imag.collections) >= 2
    plt.close(fig_real)
    plt.close(fig_imag)


def test_pt3_ratio_plot_draws_fit_overlay():
    tau_dict = {4: np.arange(5), 6: np.arange(7)}
    ratio_real = {
        4: gv.gvar(np.linspace(0.8, 1.0, 5), np.full(5, 0.05)),
        6: gv.gvar(np.linspace(0.85, 1.05, 7), np.full(7, 0.05)),
    }
    ratio_imag = {
        4: gv.gvar(np.linspace(0.1, 0.2, 5), np.full(5, 0.02)),
        6: gv.gvar(np.linspace(0.12, 0.22, 7), np.full(7, 0.02)),
    }
    p = {
        "E0": gv.gvar(0.4, 0.02),
        "dE1": gv.gvar(0.3, 0.02),
        "z0": gv.gvar(1.1, 0.05),
        "z1": gv.gvar(0.4, 0.04),
        "O00_re": gv.gvar(0.8, 0.05),
        "O01_re": gv.gvar(0.2, 0.04),
        "O11_re": gv.gvar(0.1, 0.03),
        "O00_im": gv.gvar(0.1, 0.02),
        "O01_im": gv.gvar(0.04, 0.01),
        "O11_im": gv.gvar(0.02, 0.01),
    }

    (fig_real, ax_real), (fig_imag, ax_imag) = pt3_ratio_plot(
        tau_dict,
        ratio_real,
        ratio_imag,
        fit_result=FitResult(p),
        fit_tau_cut=1,
        Lt=32,
    )

    assert ax_real.get_legend() is not None
    assert ax_imag.get_legend() is not None
    assert len(ax_real.collections) >= 4
    assert len(ax_imag.collections) >= 4
    plt.close(fig_real)
    plt.close(fig_imag)


def test_fh_plot_draws_fit_overlay():
    tsep_ls = [4, 6, 8]
    fh_real = gv.gvar([0.8, 0.82, 0.84], [0.05, 0.05, 0.05])
    fh_imag = gv.gvar([0.1, 0.11, 0.12], [0.02, 0.02, 0.02])
    p = {
        "E0": gv.gvar(0.4, 0.02),
        "O00_re": gv.gvar(0.8, 0.05),
        "O00_im": gv.gvar(0.1, 0.02),
    }

    (fig_real, ax_real), (fig_imag, ax_imag) = fh_plot(
        tsep_ls,
        fh_real,
        fh_imag,
        fit_result=FitResult(p),
    )

    assert ax_real.get_legend() is not None
    assert ax_imag.get_legend() is not None
    assert len(ax_real.collections) >= 2
    assert len(ax_imag.collections) >= 2
    plt.close(fig_real)
    plt.close(fig_imag)


def test_ff_ratio_plot_draws_data():
    tau_dict = {4: np.arange(5), 6: np.arange(7)}
    ratio_real = {
        4: gv.gvar(np.linspace(0.8, 1.0, 5), np.full(5, 0.05)),
        6: gv.gvar(np.linspace(0.85, 1.05, 7), np.full(7, 0.05)),
    }

    fig, ax = ff_ratio_plot(tau_dict, ratio_real)

    assert fig is not None
    assert ax.get_xlabel()
    assert len(ax.collections) >= 2


def test_ff_sum_plot_draws_data():
    tsep_ls = [4, 6, 8]
    sum_real = gv.gvar([0.8, 0.9, 1.0], [0.05, 0.05, 0.05])

    fig, ax = ff_sum_plot(tsep_ls, sum_real)

    assert fig is not None
    assert ax.get_ylabel()
    assert len(ax.collections) >= 1


def test_ff_joint_plot_draws_ratio_and_sum():
    tau_dict = {4: np.arange(5), 6: np.arange(7)}
    ratio_real = {
        4: gv.gvar(np.linspace(0.8, 1.0, 5), np.full(5, 0.05)),
        6: gv.gvar(np.linspace(0.85, 1.05, 7), np.full(7, 0.05)),
    }

    (fig_ratio, ax_ratio), (fig_sum, ax_sum) = ff_joint_plot(tau_dict, ratio_real)

    assert fig_ratio is not None
    assert fig_sum is not None
    assert ax_ratio.get_xlabel()
    assert ax_sum.get_xlabel()


def test_pt2_plot_draws_fit_overlay():
    Lt = 32
    t = np.arange(Lt)
    params = {"E0": 0.45, "dE1": 0.55, "z0": 1.1, "z1": 0.45}
    mean = pt2_re_fcn(t, params, Lt, nstate=2)
    pt2_avg = gv.gvar(mean, np.maximum(0.02 * mean, 1e-6))

    prior = gv.BufferDict()
    prior["E0"] = gv.gvar(0.45, 0.2)
    prior["log(dE1)"] = gv.gvar(np.log(0.55), 0.4)
    prior["z0"] = gv.gvar(1.1, 0.4)
    prior["z1"] = gv.gvar(0.45, 0.4)
    fit = pt2_fit(pt2_avg, 3, 12, Lt, prior=prior, label="Fit")

    (fig_c2, ax_c2), (fig_meff, ax_meff) = pt2_plot(
        [pt2_avg],
        trange=(2, 14),
        fit_results=fit,
        fit_tmin=3,
        fit_tmax=12,
        fit_label="Fit",
    )

    assert ax_c2.get_yscale() == "log"
    assert ax_c2.get_legend() is not None
    assert ax_meff.get_legend() is not None
    assert len(ax_c2.lines) >= 2
    assert len(ax_meff.lines) >= 2

    plt.close(fig_c2)
    plt.close(fig_meff)
