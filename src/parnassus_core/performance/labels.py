VAR_LABELS = {
    # Event variables
    "npart": r"$N_{\mathrm{part}}$",
    "njet": r"$N_{\mathrm{jet}}$",
    "ht": r"$H_{T}$",
    "met_x": r"$E_{x}^{\mathrm{miss}}$",
    "met_y": r"$E_{y}^{\mathrm{miss}}$",
    # Particle variables
    "pt": r"$p_{T}$",
    "eta": r"$\eta$",
    "phi": r"$\phi$",
    "mass": r"$m$",
    "vx": r"$v_{x}$",
    "vy": r"$v_{y}$",
    "vz": r"$v_{z}$",
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
}

RESIDUAL_VAR_LABELS = {
    # Event variables
    "npart_res": r"$N_{\mathrm{part}}^{\mathrm{reco}} - N_{\mathrm{part}}^{\mathrm{truth}}$",
    "njet_res": r"$N_{\mathrm{jet}}^{\mathrm{reco}} - N_{\mathrm{jet}}^{\mathrm{truth}}$",
    "ht_res": r"Residual $H_T$",
    "met_x_res": r"Residual $E_{x}^{\mathrm{miss}}$",
    "met_y_res": r"Residual $E_{y}^{\mathrm{miss}}$",
    # Particle variables
    "pt_res": r"$p_{T}$",
    "eta_res": r"$\eta$",
    "phi_res": r"$\phi$",
    "mass_res": r"$m$",
    "vx_res": r"$v_{x}$",
    "vy_res": r"$v_{y}$",
    "vz_res": r"$v_{z}$",
    # Substructure variables
    "d2_res": r"$\ln D_2$",
    "c2_res": r"$C_2$",
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
}
