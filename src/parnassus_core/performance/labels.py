VAR_LABELS = {
    # Event variables
    "npart": r"$N_{\mathrm{part}}$",
    "njet": r"$N_{\mathrm{jet}}$",
    "ht": r"$H_{T}$",
    "met_x": r"$E_{x}^{\mathrm{miss}}$",
    "met_y": r"$E_{y}^{\mathrm{miss}}$",
    "ht_p": r"$\sum p$",
    "Pxh": r"$H_{x}$",
    "Pyh": r"$H_{y}$",
    # Particle variables
    "pt": r"$p_{T}$",
    "p": r"$p$",
    "eta": r"$\eta$",
    "theta": r"$\theta$",
    "phi": r"$\phi$",
    "mass": r"$m$",
    "vx": r"$v_{x}$",
    "vy": r"$v_{y}$",
    "vz": r"$v_{z}$",
    "beta": r"$\beta$",
    # Impact parameter variables
    "d0": r"$d_{0}$",
    "d0Error": r"$\sigma_{d_{0}}$",
    "d0Sig": r"$d_{0}/\sigma_{d_{0}}$",
    "z0": r"$z_{0}$",
    "z0Error": r"$\sigma_{z_{0}}$",
    "z0Sig": r"$z_{0}/\sigma_{z_{0}}$",
    "ip3d": r"IP3D",
    "ip3dError": r"IP3D Error",
    "ip3dSig": r"IP3D Significance",
    # Substructure variables
    "d2": r"$\ln D_2$",
    "c2": r"$C_2$",
    # DIS kinematics
    "Q2": r"$Q^2$",
    "xB": r"$x_B$",
    "W": r"$W$",
    "y": r"$y$",
    "nu": r"$\nu$",
}

RESIDUAL_VAR_LABELS = {
    # Event variables
    "npart_res": r"$N_{\mathrm{part}}^{\mathrm{reco}} - N_{\mathrm{part}}^{\mathrm{truth}}$",
    "njet_res": r"$N_{\mathrm{jet}}^{\mathrm{reco}} - N_{\mathrm{jet}}^{\mathrm{truth}}$",
    "ht_res": r"Residual $H_T$",
    "met_x_res": r"Residual $E_{x}^{\mathrm{miss}}$",
    "met_y_res": r"Residual $E_{y}^{\mathrm{miss}}$",
    "ht_p_res": r"Residual $\sum p$",
    "Pxh_res": r"Residual $H_{x}$",
    "Pyh_res": r"Residual $H_{y}$",
    # Particle variables
    "pt_res": r"$p_{T}$",
    "p_res": r"$p$",
    "eta_res": r"$\eta$",
    "theta_res": r"$\theta$",
    "phi_res": r"$\phi$",
    "mass_res": r"$m$",
    "vx_res": r"$v_{x}$",
    "vy_res": r"$v_{y}$",
    "vz_res": r"$v_{z}$",
    "beta_res": r"$\beta$",
    # Substructure variables
    "d2_res": r"$\ln D_2$",
    "c2_res": r"$C_2$",
    # DIS kinematics (relative residuals)
    "Q2_res": r"Rel. Residual $Q^2$",
    "xB_res": r"Rel. Residual $x_B$",
    "W_res": r"Rel. Residual $W$",
    "y_res": r"Rel. Residual $y$",
    "nu_res": r"Rel. Residual $\nu$",
}

COLORS = [
    "#ff7f0e",
    "#2ca02c",
    "#d62728",
    "#9467bd",
    "#8c564b",
    "#e377c2",
    "#7f7f7f",
    "#bcbd22",
    "#17becf",
    "#1f77b4",
]

SAMPLE_LABELS = {
    "h4lep": r"$H\to 4\l$",
    "ttbar": r"$t\bar{t}$",
    "qcd_470": r"QCD $470$",
    "qcd_600": r"QCD $600$",
    "qcd15to3000": r"QCD $15-3000$",
    "dyjets": r"$Z/\gamma^* \to \ell\ell$",
    # CLAS12 RGA
    "clasdis": r"CLAS12 RGA",
    "dvcs": r"dvcs",
}
